import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time
import pickle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
#import shap
from joblib import dump
import torch.jit
import matplotlib.pyplot as plt
import copy
PRINTTIME = False

current_directory = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = current_directory+"\\ML_Modelle"
#MODEL_SAVE_PATH = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\ML_Modelle"


def train_NN(params,X_train, X_test ,y_train ,y_test):
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    y_train = y_train.T
    y_test = y_test.T
    #print(f"training with input size {input_size} and output {output_size}\n\n")
    #X_train = X_train.T
    #X_test = X_test.T

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"device: {device}")
    #print(f"predicitons\n{predictions}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    #scaler = StandardScaler()
    #scaler.fit(ml_validation_vector_aircut)
    #scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Parameters
    hidden_size = 128
    learning_rate = 0.001
    #batch_size = 512 #not in use
    #num_epochs = 1000
    num_epochs = params.num_epochs

    #'''
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) #nn.Linear(144, hidden_size
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc2a = nn.Linear(hidden_size, hidden_size) #added4test
            self.fc2b = nn.Linear(hidden_size, hidden_size) #added4test
            self.fc2c = nn.Linear(hidden_size, hidden_size) #added4test
            self.fc2d = nn.Linear(hidden_size, hidden_size) #added4test
            self.fc2e = nn.Linear(hidden_size, hidden_size) #added4test
            #self.fc2f = nn.Linear(hidden_size, hidden_size) #added4test
            #self.fc2g = nn.Linear(hidden_size, hidden_size) #added4test
            #self.fc2h = nn.Linear(hidden_size, hidden_size) #added4test
            self.fc3 = nn.Linear(hidden_size, output_size) #nn.Linear(hidden_size, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc2a(x)) #added4test
            x = self.relu(self.fc2b(x)) #added4test
            x = self.relu(self.fc2c(x)) #added4test
            x = self.relu(self.fc2d(x)) #added4test
            x = self.relu(self.fc2e(x)) #added4test
            #x = self.relu(self.fc2f(x)) #added4test
            #x = self.relu(self.fc2g(x)) #added4test
            #x = self.relu(self.fc2h(x)) #added4test

            x = self.fc3(x)
            return x
    #'''

    model = Net().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    training_start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % int(num_epochs/10) == 0:
            noop = 1
            #print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    #print(f"Finished training in: {time.time()-training_start_time} seconds")
    # Evaluate the model on the test data
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        #print(f"Test Loss: {test_loss.item()}")

    # Make predictions using the trained model
    predictions = model(X_test).detach().cpu().numpy()
    #print(f"finished train_nn in trf")
    return predictions

def save_rf_model_as_pkl(model, filename):
    dump(model, filename)

def train_RF_Normal(model_string,params, X_train, X_test, y_train, y_test):
    y_train = y_train.reshape(-1)
    # Initialize the Scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled =scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=10,
                               max_depth=20,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               n_jobs=-1)
    # Training the model
    rf.fit(X_train_scaled, y_train)
    # Make predictions
    predictions = rf.predict(X_test_scaled)
    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = rf.predict(X_test_scaled)
            totaltime += time.time()-startpredict
        print(f"RF_Normal took {totaltime} seconds")
    save_rf_model_as_pkl(rf, MODEL_SAVE_PATH+"\\"+model_string+".pkl")
    #print(f"saved model")
    return predictions

def train_RF_Cheap(model_string,params, X_train, X_test, y_train, y_test):
    y_train = y_train.reshape(-1)
    # Initialize the Scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled =scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=10,
                               max_depth=3,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               n_jobs=-1)
    # Training the model
    rf.fit(X_train_scaled, y_train)
    # Make predictions

    predictions = rf.predict(X_test_scaled)
    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = rf.predict(X_test_scaled)
            totaltime += time.time()-startpredict
        print(f"RF_Cheap took {totaltime} seconds")
    
    save_rf_model_as_pkl(rf, MODEL_SAVE_PATH+"\\"+model_string+".pkl")
    return np.array(predictions)

class Net_NN_Normal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net_NN_Normal, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2a = nn.Linear(hidden_size, hidden_size)
        self.fc2b = nn.Linear(hidden_size, hidden_size)
        self.fc2c = nn.Linear(hidden_size, hidden_size) 
        self.fc2d = nn.Linear(hidden_size, hidden_size) 
        self.fc2e = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size) 
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2a(x))
        x = self.relu(self.fc2b(x))
        x = self.relu(self.fc2c(x))
        x = self.relu(self.fc2d(x))
        x = self.relu(self.fc2e(x))
        x = self.fc3(x)
        return x

def train_NN_Normal(model_string,params,X_train, X_val, X_test ,y_train , y_val, y_test):
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    # Parameters
    hidden_size = 128
    learning_rate = 0.001
    num_epochs = 1000
    #print(f"input_size = {input_size}, hidden_size = {hidden_size}, output_size = {output_size}")
    
    #model = Net.to(device)
    model = Net_NN_Normal(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For storing losses
    training_losses = []
    test_losses = []
    val_losses = []
    best_loss = float('inf')
    patience = 50
    trigger_times = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # Record training loss
        training_losses.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            trigger_times = 0
        else:
            trigger_times += 1

        if trigger_times >= patience:
            #print(f"Early stopping triggered at epoch {epoch+1}. Best Validation Loss: {best_loss} Best epoch: {best_epoch}")
            model.load_state_dict(best_model)
            break  # Early stopping

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        if (epoch+1) % int(num_epochs/10) == 0:
            noop = 1
    torch.save(model.state_dict(), MODEL_SAVE_PATH+"\\"+model_string+".pt")
    #plt.figure(figsize=(10, 5))
    #plt.plot(training_losses, label='Training Loss')
    #plt.plot(test_losses, label='Test Loss')
    #plt.plot(val_losses, label='Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Loss Curves')
    #plt.legend()
    #plt.show()
    ###model_scripted = torch.jit.script(model)  # Export to TorchScript
    #model_scripted.save(MODEL_SAVE_PATH+"\\"+model_string+".pt")  # Save
    # Evaluate the model on the test data
    #with torch.no_grad():
    #    test_outputs = model(X_test)
    #    test_loss = criterion(test_outputs, y_test)
    #    #print(f"Test Loss: {test_loss.item()}")

    # Make predictions using the trained model

    
    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = model(X_test)#.detach().cpu().numpy()
            totaltime += time.time()-startpredict
        print(f"NN_Normal took {totaltime} seconds")
    predictions = model(X_test).detach().cpu().numpy()
    return predictions

def train_NN_Normal_OLD(model_string,params,X_train, X_test ,y_train ,y_test):
    train_loss_values = []  # List to store training loss values
    val_loss_values = []    # List to store validation loss values
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    y_train = y_train.T
    y_test = y_test.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    # Parameters
    hidden_size = 128
    learning_rate = 0.001
    num_epochs = 1000
    #print(f"input_size = {input_size}, hidden_size = {hidden_size}, output_size = {output_size}")
    
    #model = Net.to(device)
    model = Net_NN_Normal(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss_values.append(loss.item())

        if (epoch+1) % int(num_epochs/10) == 0:
            noop = 1
        trf_progress = (epoch/num_epochs) * 100
    model.eval()  # Set the model to evaluation mode before tracing
    torch.save(model.state_dict(), MODEL_SAVE_PATH+"\\"+model_string+".pt") #<-- this should be what paula needs
    #torch.save(model, MODEL_SAVE_PATH + "\\" + model_string + "_full.pt") #load with: model = torch.load(PATH_TO_SAVED_MODEL)
    
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(MODEL_SAVE_PATH+"\\"+model_string+"_jitscriptt.pt")  # Save
    example_input_batch = X_test[:1] #for tracing the model
    model_traced = torch.jit.trace(model, example_input_batch)
    model_traced.save(MODEL_SAVE_PATH + "\\" + model_string + "_jitscript.pt")
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_loss_values.append(val_loss.item())
    if (epoch+1) % int(num_epochs/10) == 0:
        noop = 1

    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = model(X_test)
            totaltime += time.time()-startpredict
        print(f"NN_Normal took {totaltime} seconds")
    predictions = model(X_test).detach().cpu().numpy()
    return predictions

def continue_training_NN(model_string, params, X_train, X_test, y_train, y_test):
    train_loss_values = []  # List to store training loss values
    val_loss_values = []    # List to store validation loss values
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    #print(f"continue_training_NN: input_size {input_size} output_size {output_size}")
    y_train = y_train.T
    y_test = y_test.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Load the pre-trained model
    hidden_size = 128
    #learning_rate = 0.001
    #num_epochs = 1000
    model = Net_NN_Normal(input_size, hidden_size, output_size).to(device)
    savestring = MODEL_SAVE_PATH + "\\" + model_string + ".pt"
    savestring = savestring.replace("Retrain_", "")
    model.load_state_dict(torch.load(savestring))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss_values.append(loss.item())

        # Optionally, add validation or progress printing here

    # Save the model state after training
    torch.save(model.state_dict(), MODEL_SAVE_PATH + "\\" + model_string + ".pt")

    # Other saving methods and evaluation code as in the original function

    predictions = model(X_test).detach().cpu().numpy()
    return predictions

def train_NN_Cheap(model_string,params,X_train, X_val, X_test ,y_train , y_val, y_test):
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    # Parameters
    hidden_size = 64
    learning_rate = 0.001
    num_epochs = 1000

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) #nn.Linear(144, hidden_size
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size) #nn.Linear(hidden_size, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    model = Net().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For storing losses
    training_losses = []
    test_losses = []
    val_losses = []
    best_loss = float('inf')
    patience = 50
    trigger_times = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # Record training loss
        training_losses.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            trigger_times = 0
        else:
            trigger_times += 1

        if trigger_times >= patience:
            #print(f"Early stopping triggered at epoch {epoch+1}. Best Validation Loss: {best_loss} Best epoch: {best_epoch}")
            model.load_state_dict(best_model)
            break  # Early stopping

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        if (epoch+1) % int(num_epochs/10) == 0:
            noop = 1
    torch.save(model.state_dict(), MODEL_SAVE_PATH+"\\"+model_string+".pt")
    #plt.figure(figsize=(10, 5))
    #plt.plot(training_losses, label='Training Loss')
    #plt.plot(test_losses, label='Test Loss')
    #plt.plot(val_losses, label='Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Loss Curves')
    #plt.legend()
    #plt.show()
    ###model_scripted = torch.jit.script(model)  # Export to TorchScript
    #model_scripted.save(MODEL_SAVE_PATH+"\\"+model_string+".pt")  # Save
    # Evaluate the model on the test data
    #with torch.no_grad():
    #    test_outputs = model(X_test)
    #    test_loss = criterion(test_outputs, y_test)
    #    #print(f"Test Loss: {test_loss.item()}")

    # Make predictions using the trained model

    
    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = model(X_test)#.detach().cpu().numpy()
            totaltime += time.time()-startpredict
        print(f"NN_Cheap took {totaltime} seconds")
    infstart = time.time()
    predictions = model(X_test).detach().cpu().numpy()
    print(f"inference took {time.time()-infstart}")
    return predictions

def train_NN_Cheap_formula_input(model_string,params,X_train, X_test ,y_train ,y_test):

    #change input to formula: I(t)~(c1+c2)*sign(x)+c3*MMR*kc+Cn*x_dot*x_2dot+c3*x_dot^2

    print(X_train.shape)
    print(X_test.shape)

    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    y_train = y_train.T
    y_test = y_test.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    # Parameters
    hidden_size = 64
    learning_rate = 0.001
    num_epochs = 1000

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) #nn.Linear(144, hidden_size
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size) #nn.Linear(hidden_size, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    model = Net().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % int(num_epochs/10) == 0:
            noop = 1
    torch.save(model.state_dict(), MODEL_SAVE_PATH+"\\"+model_string+".pt")
    ###model_scripted = torch.jit.script(model)  # Export to TorchScript
    #model_scripted.save(MODEL_SAVE_PATH+"\\"+model_string+".pt")  # Save
    # Evaluate the model on the test data
    #with torch.no_grad():
    #    test_outputs = model(X_test)
    #    test_loss = criterion(test_outputs, y_test)
    #    #print(f"Test Loss: {test_loss.item()}")

    # Make predictions using the trained model

    
    if PRINTTIME:
        totaltime = 0
        for i in range(100):
            startpredict = time.time()
            predictions = model(X_test)#.detach().cpu().numpy()
            totaltime += time.time()-startpredict
        print(f"NN_Cheap took {totaltime} seconds")
    predictions = model(X_test).detach().cpu().numpy()
    return predictions