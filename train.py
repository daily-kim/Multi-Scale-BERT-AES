import torch, configargparse
from data import load_asap_data
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel
from args import _initialize_arguments
from torch.utils.data import Dataset, DataLoader
from evaluate import lossfunc
from torch import optim
import copy

from transformers import logging
logging.set_verbosity_error()

class AsapDataset(Dataset):
    def __init__(self,documents, labels):
        self.documents = documents
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return [self.documents[idx], self.labels[idx]]


if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)

    # load data
    train = load_asap_data(args.train_file)
    valid = load_asap_data(args.valid_file)
    test = load_asap_data(args.test_file)
    train_documents, train_labels, valid_documents, valid_labels, test_documents, test_labels = [], [], [], [], [], []

    for _, text, label in train:
        train_documents.append(text)
        train_labels.append(label)
    for _, text, label in valid:
        valid_documents.append(text)
        valid_labels.append(label)
    for _, text, label in test:
        test_documents.append(text)
        test_labels.append(label)

    num_epochs = args.num_epochs
    learning_rate = args.lr
    
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = AsapDataset(train_documents, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DocumentBertScoringModel(args=args)
    bestmodel = copy.deepcopy(model)

    model.freeze_bert()
    optimizer =optim.Adam(model.parameters(), lr = learning_rate, betas=[0.9, 0.999], weight_decay=0.005)
    best_valid_qwk = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model.predict(batch)
            loss = lossfunc(outputs, batch[1],args.alpha,args.beta,args.gamma)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch loss: {:.4f}'.format(epoch_loss))
        print('-' * 10)
        if epoch % 1 == 0:
            print(f'epoch: {epoch}, loss: {epoch_loss}')
            valid_pearson, valid_qwk = model.eval_all((valid_documents, valid_labels))
            print(f'valid_pearson: {valid_pearson}, valid_qwk: {valid_qwk}')
            #early stopping
            if valid_qwk > best_valid_qwk:
                best_valid_qwk = valid_qwk
                model.save_model()
                bestmodel = copy.deepcopy(model)
                
    print(f'alpha: {args.alpha}, beta: {args.beta}, gamma: {args.gamma}')
    test_pearson, test_qwk = bestmodel.eval_all((test_documents, test_labels))
    print(f'test_pearson: {test_pearson}, test_qwk: {test_qwk}')
    # print('Finished Training')