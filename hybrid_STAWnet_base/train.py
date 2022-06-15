import torch
import numpy as np
import argparse
import time
import util
from engine import trainer

import wandb
from distutils.util import strtobool

wandb.init(project="test-hybrid", allow_val_change=True, entity="pg-test-zasn",config={
  "learning_rate": 0.001872 ,
  "epochs": 100,
  "batch_size": 32,
  "emb_length": 32,
  "dropout": 0.4158,
  "weight_decay": 0.0001457,
  "gcn_bool": True,
  "gat_bool": True,
  "gcn_blocks": 3, # ignore in sweep
  "gat_blocks": 1, 
  "aptonly": False,
  "addaptadj": True,
  "randomadj": False,
})


config = wandb.config

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gat_bool',default=config.gat_bool, type=lambda x: bool(strtobool(str(x))), help='whether to add graph attention layer')
parser.add_argument('--gcn_bool',default=config.gcn_bool, type=lambda x: bool(strtobool(str(x))), help='whether to add graph convolution layer')
parser.add_argument('--aptonly',default=config.aptonly, type=lambda x: bool(strtobool(str(x))), help='whether only use node embedding to do attention')
parser.add_argument('--noapt',default=False,help='whether not use node embedding to do attention')
parser.add_argument('--addaptadj',default=config.addaptadj, type=lambda x: bool(strtobool(str(x))), help='whether add adaptive adj')
parser.add_argument('--randomadj',default=config.randomadj, type=lambda x: bool(strtobool(str(x))), help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--emb_length',type=int,default=16,help='node embedding length')
parser.add_argument('--in_dim',type=int,default=2 ,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes, METR:207, PEMS:325')
parser.add_argument('--batch_size',type=int,default=config.batch_size,help='batch size')
parser.add_argument('--learning_rate',type=float,default=config.learning_rate,help='learning rate')
parser.add_argument('--dropout',type=float,default=config.dropout,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=config.weight_decay,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=config.epochs,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--gat_blocks',type=int,default=3,help='')
parser.add_argument('--gcn_blocks',type=int,default=1,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metrtesthybridtrain1',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

wandb.config.update({
    "gcn_bool":args.gcn_bool,
    "gat_bool":args.gat_bool,
    "aptonly":args.aptonly,
    "gat_blocks":args.gat_blocks,
    "gcn_blocks": 4 - args.gat_blocks,
    "addaptadj":args.addaptadj,
    "randomadj":args.randomadj
},allow_val_change=True);


def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, config.batch_size, config.batch_size, config.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx] # supports是adj_mx

    print(args)

    if config.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if config.aptonly:
        supports = None



    engine = trainer(scaler, config.gat_blocks, 4 - config.gat_blocks, args.in_dim, args.seq_length, args.num_nodes, args.nhid, config.dropout,
                         config.learning_rate, config.weight_decay, device, config.gat_bool, config.gcn_bool, config.addaptadj,
                         adjinit, config.aptonly, config.emb_length, args.noapt, supports=supports)


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    writer = SummaryWriter()
    for i in range(1,config.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        iter = dataloader['train_loader'].get_iterator()
        max_steps = sum(1 for _ in iter)
        iter = dataloader['train_loader'].get_iterator()
        for iter, (x, y) in enumerate(iter):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            metrics2 = {
                    "train/train_loss": train_loss[-1], 
                    "train/train_mape": train_mape[-1], 
                    "train/train_rmse": train_rmse[-1], 
                    "train/epoch": (iter + (max_steps * (i-1))) / max_steps
                    }
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                wandb.log(metrics2)
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {
            "val/val_loss": mvalid_loss, 
            "val/val_mape": mvalid_mape,
            "val/val_rmse": mvalid_rmse,
        }
        wandb.log({**metrics2, **val_metrics})

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+"_"+str(config.learning_rate)+"_"+str(config.batch_size)+"_"+str(config.dropout)+"_"+str(config.dropout)+".pth")
        writer.add_scalars('scalar/test', {'train_loss': mtrain_loss, 'val_loss': mvalid_loss}, i)
    writer.close()
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+"_"+str(config.learning_rate)+"_"+str(config.batch_size)+"_"+str(config.dropout)+"_"+str(config.dropout)+".pth"))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, _ = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    wandb.summary['test_MAE'] = np.mean(amae)
    wandb.summary['test_MAPE'] = np.mean(amape)
    wandb.summary['test_RMSE'] = np.mean(armse)
    for i in range(args.seq_length):
        my_string = 'test_horizon_{:d}_amae'
        wandb.summary[my_string.format(i+1)] = amae[i]
        my_string = 'test_horizon_{:d}_amape'
        wandb.summary[my_string.format(i+1)] = amape[i]
        my_string = 'test_horizon_{:d}_armse'
        wandb.summary[my_string.format(i+1)] = armse[i]
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+"_"+str(config.learning_rate)+"_"+str(config.batch_size)+"_"+str(config.dropout)+"_"+str(config.dropout)+".pth")

    wandb.finish()


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
