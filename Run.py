import argparse
import torch
import torch.optim as optim

from model.callback import MyCallBack
from model.data_GCN import load_data, get_data_iterator
# from model.data import load_data, get_data_iterator
from model.evaluate import metric
from model.config import Config

####模型选择
# from model.casRel import CasRel
# from model.casRel_BiLSTM import CasRel
from model.casRel_No_Attention import CasRel
# from model.casRel_No_BiLSTM import CasRel
# from model.casRel_GCN import CasRel


import torch.nn.functional as F
from fastNLP import Trainer, LossBase

seed = 226
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)#default=1
parser.add_argument('--max_epoch', type=int, default=300)#default=10
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--dataset', default='geo', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='bert-base-chinese', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
args = parser.parse_args()
con = Config(args)


# batch_size 1   22:19:38   17:26:15       15:45:38     15:07:35    13:47:51
# batch_size 2   14:58:31   12:44:36       11:57:15     11:35:42    11:28:46    0.90408
# batch_size 4                                                      10:44:39    1.01826
# batch_size 8   10:59:27   3.85810
# batch_size 16  12:00:00
class MyLoss(LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def get_loss(self, predict, target):
        mask = target['mask']

        def loss_fn(pred, gold, mask):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy(pred, gold, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        return loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
               loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
               loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
               loss_fn(predict['obj_tails'], target['obj_tails'], mask)

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(pred_dict, target_dict)
        return loss


if __name__ == '__main__':
    model = CasRel(con).to(device)
    data_bundle, rel_vocab = load_data(con.train_path, con.dev_path, con.test_path, con.rel_path)
    train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab)
    dev_data = get_data_iterator(con, data_bundle.get_dataset('dev'), rel_vocab, is_test=True)
    test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer, batch_size=con.batch_size,
                      n_epochs=con.max_epoch, loss=MyLoss(), print_every=con.period, use_tqdm=True,
                      callbacks=MyCallBack(dev_data, rel_vocab, con))
    trainer.train()
    print("-" * 5 + "Begin Testing" + "-" * 5)
    metric(test_data, rel_vocab, con, model)
