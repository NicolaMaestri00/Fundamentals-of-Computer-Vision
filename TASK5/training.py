import torch
import os
from pathlib import Path
import warnings

#--------------------------------------Define Step--------------------------------------
def build_step(model, loss_fn, optimizer, mode='train'):
    def train_step_def(x, y):
        model.train()
        probs = model(x)
        loss = loss_fn(probs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    def val_step_def(x, y):
        with torch.no_grad():
            model.eval()
            probs = model(x)
            loss = loss_fn(probs, y)
            return loss.item()
    def test_step_def(x):
        with torch.no_grad():
            model.eval()
            probs = model(x)
            return probs
    if mode == 'train':
        return train_step_def
    if mode == 'val':
        return val_step_def
    if mode == 'test':
        return test_step_def
    else:
        return train_step_def


#--------------------------------------train epoch--------------------------------------
def train_epoch(args, model, train_step, trn_loader):
    
    correct, total, loss_sum = 0, 0, 0.
    for step, data in enumerate(trn_loader):
        inputs, labels= data[0].to(args.DEVICE), data[1].to(args.DEVICE)
        
        #loss
        loss_sum += train_step(inputs, labels)
        loss = loss_sum / (step+1)
        
        #predict
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
        
        #acc
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = 100*correct / total
        
        #train_time
        args.end.record()
        torch.cuda.synchronize()
        time = args.start.elapsed_time(args.end)
        
        print('\r[{:{}d}/{}] Train {:{}d}/{} | Loss:{:8.5f} Acc:{:6.2f}% | Time: {:.2f}s'.format(
                args.epoch+1, len(str(args.EPOCHS)), args.EPOCHS, 
                step+1, len(str(len(trn_loader))), len(trn_loader),
                loss, acc, time/1000), end='')

    return loss, acc, time

#--------------------------------------val epoch--------------------------------------  
def val_epoch(args, model, loss_function, val_loader):
    with torch.no_grad():
        correct, total, val_loss_sum = 0, 0, 0.
        for step, data in enumerate(val_loader):
            inputs, labels= data[0].to(args.DEVICE).float(), data[1].to(args.DEVICE)
            
            #val_predict
            model.eval()
            pred = model(inputs)
            
            #val_loss
            val_loss_sum += loss_function(pred, labels).item()
            val_loss = val_loss_sum / (step+1)
            
            #val_acc
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100*correct / total
            
            #val_time
            args.end.record()
            torch.cuda.synchronize()
            time = args.start.elapsed_time(args.end)
            
            print('\r[{:{}d}/{}] Evaluate {:{}d}/{} | Val_loss:{:8.5f} Val_acc:{:6.2f}% | Time: {:.2f}s'.format(
                    args.epoch+1, len(str(args.EPOCHS)), args.EPOCHS, 
                    step+1, len(str(len(val_loader))), len(val_loader), 
                        val_loss, val_acc, time/1000), end='')

    return val_loss, val_acc, time

#--------------------------------------------Main---------------------------------------------
def run(args, model, loss_function, optimizer, checkpoint, trn_loader, val_loader):
    #---------------------------------------Initialization----------------------------------------
    train_step = build_step(model, loss_function, optimizer)
    loss_history, val_loss_history, acc_history, val_acc_history, time_history = [],[],[],[],[]
    args.epoch, best_loss, best_acc, stop_count = 0, 10.0, 0., 0
    #---------------------------------------load Checkpoint---------------------------------------
    if checkpoint is not None:
        args.epoch = checkpoint['epoch']+1
        best_loss = checkpoint['best_loss']
        stop_count = checkpoint['stop_count']
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        acc_history = checkpoint['acc_history']
        val_acc_history = checkpoint['val_acc_history']
        time_history = checkpoint['time_history']
        print('Back to training', model.model)
        _ = [print('[{:{}d}/{}] Loss:{:8.5f} Acc:{:6.2f}% | {}ms per step | Val_loss:{:8.5f} Val_acc:{:6.2f}% | {}ms per step'.format(
                    i+1, len(str(args.EPOCHS)), args.EPOCHS, 
                    loss_history[i], acc_history[i],
                    int(time_history[i][0]/args.num_trn_data*args.BATCH_SIZE),
                    val_loss_history[i], val_acc_history[i],
                    int(time_history[i][1]/args.num_val_data*args.BATCH_SIZE),
                    )
                ) for i in range(args.epoch)]
        print('--------------Previous point--------------')

    if not os.path.exists(Path(args.PATH+'State/'+model.model)):
        os.mkdir(args.PATH+'State/'+model.model)
    args.start = torch.cuda.Event(enable_timing=True)
    args.end = torch.cuda.Event(enable_timing=True)

    for args.epoch in range(args.epoch, args.EPOCHS):
        #------------------------------------------Train------------------------------------------
        args.start.record()
        warnings.filterwarnings('ignore')
        
        loss, acc, time = train_epoch(args, model, train_step, trn_loader)
        
        #save_train_history
        per_train_step = int(time/args.num_trn_data*args.BATCH_SIZE)
        acc_history.append(acc)
        loss_history.append(loss)
        time_history.append([time])

        warnings.filterwarnings('default')

        #----------------------------------------Evaluate-----------------------------------------
        args.start.record()
        val_loss, val_acc, time = val_epoch(args, model, loss_function, val_loader)

        #save_val_history
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        time_history[args.epoch].append(time)
        
        #---------------------------------------Checkpoint---------------------------------------
        torch.save({
            'epoch': args.epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'loss_function': loss_function,
            'stop_count': stop_count,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'acc_history': acc_history,
            'val_acc_history': val_acc_history,
            'time_history': time_history,
            'SIZE': args.SIZE,
            }, args.PATH+'State/checkpoint.pt')
        #-----------------------------------------Save Best State of Model------------------------------------------
        data_size = args.num_trn_data + args.num_val_data
        stop_count += 1

        combine_loss = args.num_val_data/data_size * val_loss_history[-1] + args.num_trn_data/data_size * loss_history[-1]
        if best_loss > combine_loss:
            torch.save(model.state_dict(), args.PATH+'State/'+model.model+'/Best_loss_state.pt')
            torch.save({
                    'loss': loss_history[-1],
                    'val_loss': val_loss_history[-1],
                    'acc': acc_history[-1],
                    'val_acc': val_acc_history[-1],
                    }, args.PATH+'State/'+model.model+'/Best_loss_history.pt')
            best_loss = combine_loss
            stop_count = 0
            
        if best_acc < val_acc_history[-1]:
            torch.save(model.state_dict(), args.PATH+'State/'+model.model+'/Best_acc_state.pt')
            torch.save({
                    'loss': loss_history[-1],
                    'val_loss': val_loss_history[-1],
                    'acc': acc_history[-1],
                    'val_acc': val_acc_history[-1],
                    }, args.PATH+'State/'+model.model+'/Best_acc_history.pt')
            best_acc = val_acc_history[-1]
            stop_count = 0
        
        print('\r[{:{}d}/{}] Loss:{:8.5f} Acc:{:6.2f}% | {}ms per step | Val_loss:{:8.5f} Val_acc:{:6.2f}% | {}ms per step'.format(
                args.epoch+1, len(str(args.EPOCHS)), args.EPOCHS, 
                loss_history[-1], acc_history[-1],
                int(time_history[-1][0]/args.num_trn_data*args.BATCH_SIZE),
                val_loss_history[-1], val_acc_history[-1], 
                int(time_history[-1][1]/args.num_val_data*args.BATCH_SIZE)))

        if stop_count >= 100:
            print('------------------------------Early stop------------------------------')
            break
    #-----------------------------------------Save Final Model and History------------------------------------------
    print('\nSaving...'.format())
    torch.save(model, args.PATH+'State/'+model.model+'/model.pt')
    torch.save(model.state_dict(), args.PATH+'State/'+model.model+'/state.pt')
    torch.save({
            'optimizer': optimizer,
            'loss_function': loss_function,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'acc_history': acc_history,
            'val_acc_history': val_acc_history,
            }, args.PATH+'State/'+model.model+'/history.pt')
    os.remove(Path(args.PATH+'State/checkpoint.pt'))
    print('Done'.format())
    print('\nLoss: {:.5f}\nAccuracy: {:.2f}%\nVal_loss: {:.5f}\nVal_accuracy: {:.2f}%'.format(
        loss_history[-1], acc_history[-1], val_loss_history[-1], val_acc_history[-1]))