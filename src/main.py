
### Authors: Tomer Gotesdyner and Yanai Levi ###

### Imports ###
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


from torch.utils.tensorboard import SummaryWriter

from seqDataset import SeqDataset
from models import CNN1DModel




def top_n_accuracy(y_true, y_pred, n=3):
    # Get the indices of the top-n predictions for each sample
    top_n_pred_indices = np.argsort(y_pred, axis=1)[:, -n:]

    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Check if the true label of each sample is in the top-n predictions
    correct_predictions = np.sum(y_true_onehot[np.arange(len(y_true))[0], top_n_pred_indices], axis=1)

    # Calculate the top-n accuracy
    top_n_acc = np.mean(correct_predictions)

    return top_n_acc   
    
def train(epoch):
    network.train() # to make sure we go through "dropout" in network
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data = torch.unsqueeze(data, dim=1)
        optimizer.zero_grad() # to make the previous gradients zero
        # forward pass
        output, embeding = network(data)

        # calculate the loss
        loss = criterion(output, target)

        # temp = loss
        # temp.to('cpu')
        # if np.isnan(temp.item()) or np.isinf(temp.item()): break
        
        # computes the gradients of the loss function
        loss.backward()
            
        # useful for preventing exploding gradients
        torch.nn.utils.clip_grad_value_(network.parameters(), 1)
        
        # update the parameters of the model
        optimizer.step()
        
        # print the loss every log_interval
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')      
    return loss


def test():
  # set the network to evaluation mode
  network.eval()
  test_loss = 0
  correct = 0
  
  with torch.no_grad(): #  avoid tracking gradients- for time consumption
    for data, target in val_loader:
      data = data.to(device)
      target = target.to(device)
      data = torch.unsqueeze(data, dim=1) # add a dimension of size one at index 1
      
      # forward pass
      output, embeding = network(data)
 
      pred = output.data.max(1, keepdim=True)[1]
      print("guess is: ", pred.transpose(0, 1))
      print("target is: ", target)
      for index, value in enumerate(pred):          
            if target[index] == value:
                  correct += 1
    
      # correct += top_n_accuracy(target.data.cpu().numpy(), output.data.cpu().numpy(), n=1)*len(target)
            
      # correct += pred.eq(target.data.view_as(pred)).sum()
        
      test_loss += criterion(output, target)
        
  test_loss /= (len(val_loader.dataset)/batch_size_test)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}\n'.format(
    test_loss))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))

  
  return test_loss, 100*correct / len(val_loader.dataset)



# load the data
dataset = SeqDataset(pathToData = "data/10_classes_new_data.fasta", metaDataFile = "data/download_with_class_id_10_classes_new_data")
# split the data to train and validation randomly
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),int(len(dataset)-int(len(dataset)*0.8))])
# create the data loaders
batch_size_test = 4
batch_size_train = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the model
input_size = 13000
num_of_classes = dataset.getNumOfClasses()
model = CNN1DModel(input_size, num_of_classes, network_size=16)

# Print the model architecture
print(model)

# network = CNN10(batch_size_val=batch_size_test, batch_size_train=batch_size_train, num_of_classes=num_of_classes, device=device).to(device)
network = model.to(device)


log_interval = 10
learning_rate = 10e-5

# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                       momentum=momentum)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr = learning_rate, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss()


n_epochs = 200
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


writer = SummaryWriter(log_dir='logs/normImage_kernel5_10000data_10classes_200epochs_top1accuracy_10e-5lr_newData_newPreprocessing_batch4_networkSize16_fixed_test')

# perform the training, with validation
#test()
for epoch in range(1, n_epochs + 1):
  train_loss = train(epoch)
  writer.add_scalar('Loss/train', train_loss, epoch)
  test_loss,accuracy = test()
  writer.add_scalar('Loss/test', test_loss, epoch)
  writer.add_scalar('Accuracy/test', accuracy, epoch)