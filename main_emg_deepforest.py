from deepforest import CascadeForestClassifier
from argparse import ArgumentParser
from torch.utils.data import Dataset,DataLoader

import numpy as np
import torch
import os
import glob
from sklearn.metrics import accuracy_score


class EmgDataset(Dataset):
    def __init__(self,samples_list):
        self.files=[]
        for sample in samples_list:
            # tem=os.listdir(sample)
            # tem.sort(key=int)
            self.files+=[os.path.join(sample,"data.npz")]
    def __getitem__(self,idx):
        sample_path = self.files[idx]
        with np.load(sample_path) as f:
            x=f["x"]
            y=f["y"]
            return {
                "x":x,
                "y":y
            }
    def __len__(self):
        return len(self.files)

def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _gen_dsuk_fold(args,num):
    # choose subject(num+1) for test
    class_num = args.label_num
    SAVE_PATH = args.data_path
    subj_file_list = os.listdir(os.path.join(SAVE_PATH))
    subj_file_list.sort()
    ways=3  # 1.just use session1  2.just ues session2  3.use both
    train_samples = []
    test_samples = []
    test_samples1 = []
    test_samples2 = []
    np.random.seed(0)

    # different subject
    if ways==1:
        for subj_file_name in subj_file_list:
            for label in range(class_num):
                if subj_file_name==f"subject_{num}":
                    break
                temp_train = glob.glob(os.path.join(SAVE_PATH, subj_file_name, f"session_{args.session_i}", str(label), "*"))
                np.random.shuffle(temp_train)
                train_samples += temp_train

        for label in range(class_num):
            temp_test = glob.glob(os.path.join(SAVE_PATH, f"subject_{num}", f"session_{args.session_i}", str(label), "*"))
            np.random.shuffle(temp_test)
            test_samples += temp_test

    if ways==2:
        for subj_file_name in subj_file_list:
            for label in range(class_num):
                if subj_file_name == f"subject_{num}":
                    break
                temp_train = glob.glob(
                    os.path.join(SAVE_PATH, subj_file_name, f"session_{args.session_i+1}", str(label), "*"))
                np.random.shuffle(temp_train)
                train_samples += temp_train

        for label in range(class_num):
            temp_test = glob.glob(
                os.path.join(SAVE_PATH, f"subject_{num}", f"session_{args.session_i+1}", str(label), "*"))
            np.random.shuffle(temp_test)
            test_samples += temp_test

    if ways==3:
        for subj_file_name in subj_file_list:
            for label in range(class_num):
                if subj_file_name == f"subject_{num}":
                    break
                temp_train1 = glob.glob(os.path.join(SAVE_PATH, subj_file_name, f"session_{args.session_i}", str(label), "*"))
                temp_train2 = glob.glob(os.path.join(SAVE_PATH, subj_file_name, f"session_{args.session_i+1}", str(label), "*"))
                np.random.shuffle(temp_train1)
                np.random.shuffle(temp_train2)
                train_samples += temp_train1+temp_train2

        for label in range(class_num):
            temp_test1 = glob.glob(os.path.join(SAVE_PATH, f"subject_{num}", f"session_{args.session_i}", str(label), "*"))
            temp_test2 = glob.glob(os.path.join(SAVE_PATH, f"subject_{num}", f"session_{args.session_i+1}", str(label), "*"))
            np.random.shuffle(temp_test1)
            np.random.shuffle(temp_test2)
            # test_samples += temp_test1+temp_test2
            test_samples1 += temp_test1
            test_samples2 += temp_test2
        test_samples =test_samples1 + test_samples2
    train_length = len(train_samples)
    test_length = len(test_samples)
    print(f"choose subject_{num+1} to be the test")
    print(f"train length: {train_length} \n test length:{test_length} ")
    return train_samples,test_samples


if __name__ == '__main__':

    # gc.collect()

    parser=ArgumentParser()
    # data set
    parser.add_argument('--data_path', type=str, default="deepforest_top10_DF")
    parser.add_argument("--window_len",type=int,default=.25)
    parser.add_argument("--step_len",type=int,default=.125)
    parser.add_argument("--fs_emg",type=int,default=1024)
    parser.add_argument("--subject_num",type=int,default=20)

    parser.add_argument("--label_num", type=int, default=34)
    parser.add_argument("--session_i", type=int, default=0)
    # train parameters
    parser.add_argument("--estimator_number", type=int, default=2)
    parser.add_argument("--tree_number", type=int, default=100)
    parser.add_argument("--max_layer_number", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    # process
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)


    opts = parser.parse_args()
    setattr(opts, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    set_random_seed(0)

    result=np.zeros(opts.subject_num)
    for num in range(opts.subject_num):

        model = CascadeForestClassifier(max_layers=opts.max_layer_number, n_estimators=opts.estimator_number,
                                        n_trees=opts.tree_number, random_state=1, n_jobs=-1, backend="sklearn",
                                        use_predictor=True)

        train_samples,test_samples = _gen_dsuk_fold(opts,num)
        train_dataset = EmgDataset(samples_list=train_samples)
        test_dataset = EmgDataset(samples_list=test_samples)

        train_data_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,pin_memory=True, shuffle=True, drop_last=False)
        test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,pin_memory=True, shuffle=False, drop_last=False)

        train_label=[]
        train_data=[]
        y_true_list = []
        y_prod_list = []

        print("begin to train")

        for idx,batch in enumerate(train_data_loader):
            train_y = batch["y"].type(torch.long)
            train_x = batch["x"].type(torch.float32)
            train_label.append(train_y)
            train_data.append(train_x)
        label_feed=np.hstack(train_label)
        data_feed=np.vstack(train_data)
        model.fit(data_feed,label_feed)


        print(f"subject_{num} to be test")

        for idx,batch in enumerate(test_data_loader):
            test_y =batch["y"].type(torch.long)
            test_x =batch["x"].type(torch.float32)
            y_true_list.append(test_y)
            y_prod_list.append(test_x)

        y_true = np.hstack(y_true_list)
        y_prod = np.vstack(y_prod_list)
        y_pred =model.predict(y_prod)
        acc = accuracy_score(y_true, y_pred) * 100
        result[num]=acc
        print("\nTesting Accuracy: {:.3f} %".format(acc))
    print(f"mean:{result.mean()} \n std:{result.std()}")

    # for i in range(model.n_layers_):
    #     feature=model.get_layer_feature_importances(i)
    #     pathlib.Path(f"D:\TL\deepforest\layer_feature\\tree_{opts.tree_number}").mkdir(parents=True, exist_ok=True)
    #     np.savez(os.path.join(f"D:\TL\deepforest\layer_feature\\tree_{opts.tree_number}", f"feature_{i}.npz"), feature=feature)
    # model.save(f"model\\tree_{opts.tree_number}")


