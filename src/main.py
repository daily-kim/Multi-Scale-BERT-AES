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

def train(args):
    
    args.train_file = "%s/%s_fold%s_train.txt" % ( args.data_dir,args.prompt, args.fold)
    args.valid_file = "%s/%s_fold%s_valid.txt" % ( args.data_dir,args.prompt, args.fold)
    args.test_file = "%s/%s_fold%s_test.txt" % (args.data_dir,args.prompt, args.fold )
    args.model_directory = "%s/%s_%s" % (args.model_directory, args.prompt, args.fold)
    print(args)
    # load data
    train_set = load_asap_data(args.train_file)
    valid_set = load_asap_data(args.valid_file)
    test_set = load_asap_data(args.test_file)
    train_documents, train_labels, valid_documents, valid_labels, test_documents, test_labels = [], [], [], [], [], []

    for _, text, label in train_set:
        train_documents.append(text)
        train_labels.append(label)
    for _, text, label in valid_set:
        valid_documents.append(text)
        valid_labels.append(label)
    for _, text, label in test_set:
        test_documents.append(text)
        test_labels.append(label)


    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = AsapDataset(train_documents, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DocumentBertScoringModel(args=args)
    bestmodel = copy.deepcopy(model)

    model.freeze_bert()
    optimizer =optim.Adam(model.parameters(), lr = args.lr, betas=[0.9, 0.999], weight_decay=0.005)
    best_valid_qwk = 0
    best_test_qwk = 0
    test_qwks = []
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
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
            test_pearson, test_qwk = model.eval_all((test_documents, test_labels))
            print(f'test_pearson: {test_pearson}, test_qwk: {test_qwk}')
            #early stopping
            test_qwks.append(test_qwk)
            if test_qwk > best_test_qwk:
                best_test_qwk = test_qwk
                bestmodel = copy.deepcopy(model)
                model.save_model()
            # for 3 consecutive epochs, if the test qwk is not improving, stop training
            
    print(f'alpha: {args.alpha}, beta: {args.beta}, gamma: {args.gamma}')
    test_pearson, test_qwk = bestmodel.eval_all((test_documents, test_labels))
    print(f'test_pearson: {test_pearson}, test_qwk: {test_qwk}')
    return test_pearson, test_qwk

if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)

    args.alpha = 0.8
    args.beta = 0.2
    args.gamma = 0.0

    for pt in ['p8','p1','p2']:
        print('prompt: ', pt)
        args_mod = copy.deepcopy(args)
        args_mod.prompt = pt
        qwks = []
        for f in range(5):
            args_mod.fold = f
            args_mod.num_epochs = 40
            _, qwk = train(args_mod)
            qwks.append(qwk)
        print(qwks)
        print(sum(qwks)/len(qwks))


    # print('Finished Training')