##### Identification system #####
## Training:
# raw data -> feature -> database

## Testing:
# raw data -> feature -> calculate error -> identify class

import numpy as np
from scipy import io

#%% Data loading
mat1 = io.loadmat("Training.mat")
mat2 = io.loadmat("Testing.mat")

train_input = mat1.get("X") # training data
train_label = np.squeeze(mat1.get("Y")) # label
test_input = mat2.get("X") # testing data
test_label = np.squeeze(mat2.get("Y")) # label

# Set dimensions as (Event, sensor, time)
train_input=train_input.transpose(2, 1, 0)
test_input=test_input.transpose(2, 1, 0)

# Get sizes
num_event = np.size(train_input,0)
num_sensor = np.size(train_input,1)
num_sample = np.size(train_input,2)
num_eventtype = train_label[-1]

#%% Feature extraction: PCA
# Transform each sensor's data in each event to k PCs
k = 3
num_train = np.size(train_input,0)
train_feature=np.zeros((num_train,num_sensor,k))
train_eig_vecs=np.zeros((num_sensor,num_sample,k))

for sensor in range(num_sensor): 
    X=np.zeros((num_sample,num_train))
    X=np.transpose(train_input[:,sensor,:])

    # Normalize
    for i in range(num_sample):
        X[i,:]=(X[i,:]-X[i,:].mean())/X[i,:].std()
        
    # Get eigenvectors
    K=np.cov(X)
    eig_vals, eig_vecs=np.linalg.eig(K)
    
    # Sort by eigenvalue
    sorted_index=np.argsort(eig_vals)[::-1]
    sorted_eig_vecs=np.real(eig_vecs[:,sorted_index])
    train_eig_vecs[sensor]=sorted_eig_vecs[:,:k]
    
    L=np.zeros((k,num_train))
    for i in range(k):
        pc=sorted_eig_vecs[:,i]
        L[i,:]=pc.T@X
    
    train_feature[:,sensor,:]=L.T

# Test features
num_test = np.size(test_input,0)
test_feature=np.zeros((num_test,num_sensor,k))

for s in range(num_sensor):
    Y=np.zeros((num_test,num_sample))
    Y=test_input[:,s,:]
    
    # Normalize
    for i in range(num_sample):
        Y[:,i]=(Y[:,i]-Y[:,i].mean())/Y[:,i].std()
    
    # Projection on the training eigenvalues
    localeig=train_eig_vecs[s,:,:]
    F=Y@localeig
    test_feature[:,s,:]=F

#%% Classification: MSE

classified_label=np.zeros((num_test))
iden_rate=np.zeros((2,num_eventtype+1), dtype=int)
total_correct = 0
for event in range(num_test):
    sample=test_feature[event,:,:]
    
    # Compare with each training data, find the data with the lowest MSE
    lowest_mse = float("inf")
    for i in range(num_train):
        template=train_feature[i,:,:]
        mse=((sample-template)**2).mean(axis=None)
        if mse < lowest_mse:
            lowest_mse = mse
            bestsuited = i
            
    predicted = train_label[bestsuited]
    classified_label[event] = train_label[bestsuited]
    
    # Check true label
    if(predicted == test_label[event]):
        total_correct += 1
        iden_rate[0,test_label[event]]+=1
    else:
        iden_rate[1,test_label[event]]+=1

total_accuracy = total_correct/num_test
for i in range(1,num_eventtype+1):
    correct = iden_rate[0,i]
    total = correct + iden_rate[1,i]
    accuracy = round(correct / total, 3)
    print(f"Accuracy of event {i}: {accuracy} ({correct} out of {total})")

print(f"\nTotal Accuracy: {round(total_accuracy, 3)}")

#%% Classification: SFS + MSE
sensoracc = np.zeros((num_sensor,num_sensor))

selectedsensors = []
stage = 0
running = True
accofthestage=0
while running: # For each stage

    # For each sensor which has not yet been chosen, add the sensor into selected sensors and predict
    for s in [i for i in range(num_sensor) if i not in selectedsensors]:
        total_correct = 0
        localsensor = selectedsensors + [s]
        for event in range(num_test):
            sample = test_feature[event,localsensor,:]
            
            # Compare with each training data, but only using selected sensors
            lowest_mse = float("inf")
            for i in range(num_train):
                template = train_feature[i,localsensor,:]
                mse=((sample-template)**2).mean(axis=None)
                if mse<lowest_mse:
                    lowest_mse = mse
                    bestsuited = i
                    
            if(train_label[bestsuited] == test_label[event]):
                total_correct += 1        

        sensoracc[stage,s] = total_correct/num_test

    # Find adding which sensor would result in the highest accuracy, and then add it into the selected list
    counter = 0
    maxr = 0
    for r in sensoracc[stage,:]:
        if r > maxr:
            maxr = r
            chosensen = counter
        counter += 1

    selectedsensors.append(chosensen) 
    if maxr < accofthestage:
        running = False
        print(f"\nFinal Accuracy: {round(accofthestage, 3)}")

    else:
        accofthestage = maxr
        stage += 1

        print(f"\nStage {stage} accuracy: {round(accofthestage, 3)}")
        print("Used sensors: ",end="")
        for s in selectedsensors:
            print(s,end=" ")
        print(" ")

