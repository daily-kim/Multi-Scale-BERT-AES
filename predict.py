import torch, configargparse
from data import load_asap_data
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel
from args import _initialize_arguments

if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)
    print(args)

    # load data
    test = load_asap_data(args.test_file)

    test_documents, test_labels = [], []
    for _, text, label in test:
        test_documents.append(text)
        test_labels.append(label)

    print("sample number:", len(test_documents))
    print("label number:", len(test_labels))
    model = DocumentBertScoringModel(args=args)
    model.eval_all((test_documents, test_labels))
