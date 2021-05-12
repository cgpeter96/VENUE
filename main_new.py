import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim  import Adam
import json
import argparse
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from itertools import chain
from collections import defaultdict
from misc.utils import set_logger,AverageMeter,load_embedding,write_dict2vec
from misc.evalution import evaluate_clustering,get_map
from dataloader import rank_dataset
from tensorboardX import SummaryWriter
from networks.models import TrimSyn,SingleSyn
from networks.layers import check_tensor
from networks.radam import RAdam, PlainRAdam, AdamW
from networks.losses import WeightedTripletLoss,TripletLoss,ExpTripletLoss



def train(epoch, model, optimizer, loss_function, data_loader, writer=None,dataset_output_mode="multimodal"):
    model.train()
    batch_len = len(data_loader)
    batch_loss_meter = AverageMeter()

    for idx, data in enumerate(data_loader):
        label,text, visual = None,None, None
        if dataset_output_mode=="multimodal":
            label,text, visual = data
            visual = visual.view(visual.size(0),visual.size(1),visual.size(2))
            label,text, visual = check_tensor(label),check_tensor(text), check_tensor(visual)
            output_feats = model.forward_instance(text, visual)
            loss_values = loss_function(output_feats,label)

        elif dataset_output_mode=="text":
            label,feat = data
            label,feat = check_tensor(label),check_tensor(feat)
            output_feats = model.forward_instance(feat)
            loss_values = loss_function(output_feats,label)
        elif dataset_output_mode=="visual":
            label,feat = data
            label,feat = check_tensor(label),check_tensor(feat)
            output_feats = model.forward_instance(feat)
            loss_values = loss_function(output_feats,label)
        else:
             raise Exception("dataset output type error:{}".format(dataset_output_mode))
        

        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        iteration = epoch * batch_len + idx
        batch_loss_meter.update(loss_values.item())
        if iteration % 10 == 0:
            logger.info('epoch:{} ,train/batch_loss:{},iter:{}'.format(epoch,
                                                        batch_loss_meter.avg, iteration))
        writer.add_scalar("train/batch_loss", batch_loss_meter.val, iteration)
    writer.add_scalar('train/avg_loss', batch_loss_meter.avg, epoch)
    logger.info('  >>>>TRAIN Epoch:{} ,train avg loss:{}'.format(
        epoch, batch_loss_meter.avg))
    return batch_loss_meter


def evaluate(model,dataset, dataset_output_mode,save_path='',best_result=False,loss_function=None,cluster_type="HAC",extract_all=True):
    model.eval()
    model.to('cuda')
    index2word = dataset.index2word

    # 同义词ground truth
    synset = dataset.synset
    synset_vocab = list(chain(*synset))

    # 存储需要test数据
    feat1_cal={}
    feat2_cal={}
    output_emb_dict ={}#保存为.vec
    output_dim = None

    vis_mask = True
    txt_mask = False
    if vis_mask:
        mask_vec ={}
    if txt_mask:
        text_mask_vec={}
    for word, name in enumerate(tqdm(index2word, desc="extract feat")):
        if word==0:
            continue
        if not extract_all and name not in synset_vocab:
            continue


        if dataset_output_mode=="multimodal":
            word_name, word,visual = dataset.get_vocab_tensor(word)
            visual = visual.view(visual.size(0),visual.size(1))
            output_feats = model.forward_instance(check_tensor(word.unsqueeze(0)),check_tensor(visual.unsqueeze(0)))
            # 0是visual  1是text
            if name in  synset_vocab:
                if isinstance(output_feats,tuple):
                    output_dim = output_feats[0].size(-1)*2
                    feat1_cal[name]=output_feats[0].detach()
                    feat2_cal[name]=output_feats[1].detach()
                else:
                    output_dim = output_feats.size(0)
                    feat1_cal[name]=output_feats.detach()
                
            # masking vector 
            if vis_mask:
                # print(model.backbone.img_mask.mask_vec.shape)
                mask_vec[word_name]= model.backbone.img_mask.mask_vec.reshape(-1)
                # print(mask_vec[word_name].shape)
            if txt_mask:
                text_mask_vec[word_name] = model.backbone.txt_mask.mask_vec#.reshape(-1)

        elif dataset_output_mode=="visual":
            word_name ,word = dataset.get_vocab_tensor(word)
            output_feats = model.forward_instance(check_tensor(word.unsqueeze(0)))
            output_dim = output_feats.size(0)
            feat1_cal[name]=output_feats.detach()

        elif dataset_output_mode=="text":
            # word
            word_name, word  = dataset.get_vocab_tensor(word)
            output_feats = model.forward_instance(check_tensor(word))
            output_dim = output_feats.size(0)
            feat1_cal[name]=output_feats.detach()
        else:
             raise Exception("dataset output type error:{}".format(dataset_output_mode))
        
        if isinstance(output_feats,tuple):
            feat1 = output_feats[0].detach().cpu().squeeze().numpy() # visual
            feat2 = output_feats[1].detach().cpu().squeeze().numpy() # text
            output_emb_dict[name] = np.concatenate([feat1,feat2], axis=0)
        else:
            feat1 = output_feats.detach().cpu().squeeze().numpy() 
            output_emb_dict[name] = feat1

    if best_result:
        vec_path = os.path.join(save_path,"best_{}_rank.txt.vec".format(dataset_output_mode))
    else:
        vec_path =  os.path.join(save_path,"normal_{}_rank.txt.vec".format(dataset_output_mode))
    
    if extract_all:
        write_dict2vec(output_emb_dict,vec_path, output_dim)
    if vis_mask:
        mask_path =os.path.join(save_path,"mask_weight.pkl")
        write_dict2vec(mask_vec,mask_path,50)

    if txt_mask:
        text_mask_path =os.path.join(save_path,"text_mask_weight.pkl")
        write_dict2vec(text_mask_vec,text_mask_path,200)

    logger.info("===============clustering===============")
    feats1 = []#v
    feats2 = []#w
    # weight type
    if len(feat2_cal)>0:
        for name in synset_vocab:
            feats1.append(feat1_cal[name])
            feats2.append(feat2_cal[name])
        feats1 = torch.cat(feats1)
        feats2 = torch.cat(feats2)
        emb =  torch.cat([feats1, feats2],axis=1).detach().cpu().numpy()
        dis_matrix = loss_function.cal_distance(feats2,feats1)
        dis_matrix = dis_matrix.detach().cpu().numpy()
        logger.info("dis_mat:{}".format(dis_matrix[:5,:5]))
        clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         compute_full_tree=True, distance_threshold=0.27).fit(dis_matrix) 
    # embeding type
    else:
        for word in synset_vocab:
            # print(feat1_cal[word].shape)
            feat = feat1_cal[word]
            if feat.dim()>1:
                feat = feat.reshape(-1)
            feats1.append(feat)
        feats1 = torch.stack(feats1)
        emb = feats1.detach().cpu().numpy()
        print(emb.shape)
        if cluster_type =="HAC":
            clusters = AgglomerativeClustering(n_clusters=len(synset), affinity='euclidean', linkage='average',
                                         ).fit(normalize(emb))
        elif cluster_type == "matHAC":
            dis_matrix = pairwise_distances(normalize(emb))
            clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         compute_full_tree=True, distance_threshold=0.47).fit(dis_matrix) 
        elif clusters_type == "matkmeans":
            dis_matrix = pairwise_distances(normalize(emb))
            clusters = KMeans(n_clusters=len(synset),random_state=954,precompute_distances=True).fit(dis_matrix)
        elif clusters_type == "kmeans":
            clusters  = KMeans(len(synset)).fit(normalize(emb))

    # evalution metrics
    result = defaultdict(list)
    for idx,l in enumerate(clusters.labels_):
        result[l].append(synset_vocab[idx])
    pred = [v for k,v in result.items()] 
    metrics = evaluate_clustering(pred,synset)

    #mAP
    id2class =[]
    for idx,words in enumerate(synset):
        for word in words:
            id2class.append(idx)
    mAP = get_map(emb, id2class)

    metrics['map']=mAP
    for k,v in metrics.items():
        if 'predicted_clusters' in k :
            continue
        logger.info("{}:{}".format(k,v))
    return metrics

def build_model(model_type,params):
    logger.info("use {}".format(model_type))
    if model_type=="multimodal":
        # print(params["model_type"])
        return TrimSyn(img_dim=params['img_dim'],
                       txt_vocab_size=params['txt_vocab_size'],
                       txt_dim=params['txt_dim'],
                       compact_dim=params['compact_dim'],
                       output_dim=params['output_dim'],
                       op=params['op'],
                       word_embedding=params['word_embedding'],
                       model_type=params["model_type"]
                       )
    elif model_type=="visual":
        # note visual部分有效
        return SingleSyn(img_dim=params['img_dim'],
                       txt_vocab_size=params['txt_vocab_size'],
                       txt_dim=params['txt_dim'],
                       compact_dim=params['compact_dim'],
                       output_dim=params['output_dim'],
                       op=params['op'],
                       word_embedding=params['word_embedding'],
                       model_type="visual"
                       )
    elif model_type=="text":
        # note text部分有效
        return SingleSyn(img_dim=params['img_dim'],
                       txt_vocab_size=params['txt_vocab_size'],
                       txt_dim=params['txt_dim'],
                       compact_dim=params['compact_dim'],
                       output_dim=params['output_dim'],
                       op=params['op'],
                       word_embedding=params['word_embedding'],
                       model_type="text"
                       )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="split_0,new_text_data", help="split_n")
    parser.add_argument('--name', type=str, default="", help="model name")
    parser.add_argument('--epochs',type=int,default=50,help="epochs")
    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
    parser.add_argument('--output_mode',type=str,default="text",help="dataset_output_mode")
    parser.add_argument('--resume_path',type=str,default="",help="the model resume path")
    parser.add_argument('--test_mode',type=str,default="false",help="use")
    parser.add_argument('--gpu',type=str,default="",help="SET GPU")
    parser.add_argument('--w2v_type',type=str,default="word",help="use triplet")
    parser.add_argument('--log_dir',type=str,default="logs",help="log dir")
    parser.add_argument('--cluster_type',type=str,default="HAC",help="cluster type :[HAC,matHAC,kmeans,matkmeans]")
    parser.add_argument('--loss',type=str,default="triplet",help="loss function :[weight,triplet]")
    parser.add_argument('--model_type',type=str,default="",required=True,help="full,withoutTM,withoutNLA,text,visual,")
    parser.add_argument('--mu',type=float,default=0.4,help="the weighted factor")
    parser.add_argument('--compact_dim',type=int,default=200,help="the dimension of compact representation")
    parser.add_argument("--output_dim",type= int, default=512,help="the dimension of output model")


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    EPOCHS = args.epochs
    args.log_dir = os.path.join(args.log_dir,args.name)
    LOG_PATH=args.log_dir+"/runtime.log"
    

    # setting logger
    set_logger(LOG_PATH)
    # setting tensorboard
    writer = SummaryWriter(logdir=os.path.join(args.log_dir, 'board'))


    data_path = '/home/chenguang/workhome/nice_tag_data/'
    word_synset_path = 'nice_synonym_data_v2'

    options = {
        'data_path': data_path,
        'word_synset_path': word_synset_path,
        'w2v_type': args.w2v_type
    }

    word_embed_dict = {
        'wiki_unique_std':'wiki_unique_std.txt.vec',
        'mge_w2v':'new_text_data/meg_r1.5.vec'
    }

    # load_embedding
    emb_file = os.path.join(options['data_path'], options['word_synset_path'],
                      word_embed_dict[options['w2v_type']])
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(
        emb_file)
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size
    options['pretrained_embedding'] = True
    options['synset_path'] = args.data  # 'split_0'
    dataset_output_mode = args.output_mode
    word_synset_path = os.path.join(
        data_path, options['word_synset_path'], options['synset_path'])


    train_dataset = rank_dataset.BatchRankDataset(options=options,
                                                    word_synset_path=os.path.join(
                                                        word_synset_path, 'train_label.txt'),
                                                     visual_synset_path='/home/chenguang/workhome/nice_tag_data/new_wiki_data_feat2048',
                                                      #visual_synset_path='/home/chenguang/workhome/nice_tag_data/wiki_data_feat2048',
                                                     # visual_synset_path='/home/chenguang/workhome/nice_tag_data/nice_images2048',
                                                    output_mode=dataset_output_mode,
                                                    mode='train',
                                                    use_npy=True)

    test_dataset = rank_dataset.BatchRankDataset(options=options,
                                                    word_synset_path=os.path.join(
                                                        word_synset_path, 'test_label.txt'),
                                                     visual_synset_path='/home/chenguang/workhome/nice_tag_data/new_wiki_data_feat2048',
                                                      #visual_synset_path='/home/chenguang/workhome/nice_tag_data/wiki_data_feat2048',
                                                     # visual_synset_path='/home/chenguang/workhome/nice_tag_data/nice_images2048',
                                                    output_mode=dataset_output_mode,
                                                    mode='test',
                                                    use_npy=True)

    if dataset_output_mode =='text':
        collate_cn =  rank_dataset.text_collate_fn
    elif dataset_output_mode =='visual':
        collate_cn =  rank_dataset.visual_collate_fn
    else:
        collate_cn =  rank_dataset.multimodal_collate_fn
    train_loader = DataLoader(train_dataset, batch_size=64,
                            shuffle=True, num_workers=4,collate_fn=collate_cn)

    # preprea params
    # for weight loss
    model_params = {
        "img_dim":2048,
        "txt_vocab_size":vocab_size+1, # pad 0了
        "txt_dim":embed_dim,
        "compact_dim":args.compact_dim,
        "output_dim":args.output_dim,
        "op":"none", # none会输出两个结果
        "word_embedding":embedding,
        "model_type":args.model_type
    }
    
    model = build_model(dataset_output_mode, model_params)
    
    optimizer = RAdam(model.parameters(), lr=args.lr,weight_decay=0.1)
    # optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=0.1)
    # loss_function = WeightedTripletLoss(use_hardest=True,weighted_factor=0.4,margin=1.0,device="cuda")
    if args.loss == "weight":
        loss_function = WeightedTripletLoss(use_hardest=True,weighted_factor=0.4,margin=1.0,device="cuda")
    elif args.loss =="triplet":
        loss_function = TripletLoss(use_hardest=True,margin=1.0,device="cuda").to('cuda')
    #loss_function = ExpTripletLoss(use_hardest=True,margin=1.0,device="cuda").to('cuda')
    

    if args.resume_path:
        model.load_model(args.resume_path)
    if args.test_mode =="true":
       evaluate(model,test_dataset, dataset_output_mode, save_path=args.log_dir,best_result=False,loss_function=loss_function,cluster_type="HAC",extract_all=True)
       exit()

    # to gpu
    model = model.to("cuda")
    loss_function = loss_function.to("cuda")

    old_metric = 0
    change_epoch = 0 #early stop
    for epoch in range(args.epochs):
        loss_meter = train(epoch, model, optimizer, loss_function, train_loader, writer=writer,dataset_output_mode=dataset_output_mode)

        metrics = evaluate(model,test_dataset, dataset_output_mode, save_path=args.log_dir,best_result=False,loss_function=loss_function,cluster_type="HAC",extract_all=False)
        writer.add_scalar("test/ARI", metrics['ARI'] ,epoch)
        writer.add_scalar("test/FMI", metrics['FMI'] ,epoch)
        writer.add_scalar("test/NMI", metrics['NMI'] ,epoch)
        writer.add_scalar("test/homogeneity", metrics['homogeneity'] ,epoch)
        writer.add_scalar("test/completeness", metrics['completeness'] ,epoch)
        writer.add_scalar("test/v_measure_score", metrics['v_measure_score'] ,epoch)
        writer.add_scalar("test/feng_p", metrics['feng_p'] ,epoch)
        writer.add_scalar("test/feng_r", metrics['feng_r'] ,epoch)
        writer.add_scalar("test/feng_f", metrics['feng_f'] ,epoch)
        writer.add_scalar("test/mAP", metrics['map'] ,epoch)
        if True:#日常保存一些模型
            model.save_model(os.path.join(args.log_dir, 'weight', '{}-rank_model.pth'.format(epoch)))
        if old_metric < metrics["feng_f"]:
            old_metric = metrics["feng_f"]
            change_epoch = epoch
            model.save_model(os.path.join(args.log_dir, 'weight', '{}-rank_model.pth'.format(args.name)))
            with open(os.path.join(args.log_dir,"best_result.json"),"w") as fp:
                json.dump(metrics,fp)
        if epoch - change_epoch >=50:
            logger.info("early stop... at {}".format(epoch))
            break
    model.load_model(os.path.join(args.log_dir, 'weight', '{}-rank_model.pth'.format(args.name)))
    evaluate(model,test_dataset, dataset_output_mode, save_path=args.log_dir,best_result=True,loss_function=loss_function,cluster_type="HAC")

if __name__ == '__main__':
    
    main()
