
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sequence_deepctr import AttentionLayer, FullAttention, AttentionSequencePoolingLayer
import pickle

class gat_attention(nn.Module):
    def __init__(self, embedding_size, num_heads, block_shape, field_size, has_residual=True):
        super(gat_attention, self).__init__()
        self.num_heads = num_heads
        self.has_residual = has_residual

        self.A_layers = nn.ModuleDict()
        self.H_layers = nn.ModuleDict()
        self.Q_res_layers = nn.ModuleDict()
        self.S_layers_1 = nn.ModuleDict()
        self.S_layers_2 = nn.ModuleDict()
        self.Q_res_layers = nn.ModuleDict()
     
        for block in range(len(block_shape)):
            if block==0:
                input_emb_size = embedding_size
            else:
                input_emb_size = block_shape[block-1]
            self.A_layers[f'att_b{block}_0'] = nn.Linear(input_emb_size, num_heads).cuda()
            
        for block in range(len(block_shape)):
            if block==0:
                input_emb_size = embedding_size
            else:
                input_emb_size = block_shape[block-1]
            for field in range(field_size):
                 self.H_layers[f'w_b{block}_f{field}'] = nn.Linear(input_emb_size, block_shape[block], bias=False).cuda()
        
        if self.has_residual: 
            for block in range(len(block_shape)):
                if block==0:
                    input_emb_size = embedding_size
                else:
                    input_emb_size = block_shape[block-1]
                for field in range(field_size):
                    self.Q_res_layers[f'res_b{block}_f{field}'] = nn.Linear(input_emb_size, block_shape[block]).cuda()
                    
        for block in range(len(block_shape)):
            if block==0:
                input_emb_size = embedding_size
            else:
                input_emb_size = block_shape[block-1]
            self.S_layers_1[f'gsl_1_b{block}_0'] = nn.Linear(input_emb_size, 16).cuda()         
            self.S_layers_2[f'gsl_2_b{block}_0'] = nn.Linear(16, 1).cuda()

            
    def normalize(inputs, beta, gamma, epsilon=1e-8):

        input_shape = inputs.shape
        feature_size = input_shape[-1]

        mean = torch.mean(inputs, dim=-1, keepdim=True)
        variance = torch.var(inputs, dim=-1, unbiased=False, keepdim=True)
        
        beta = nn.Parameter(torch.zeros(input_emb_size))
        gamma = nn.Parameter(torch.ones(input_emb_size))

        if inputs.is_cuda:
            beta = beta.cuda()
            gamma = gamma.cuda()

        normalized = (inputs - mean) / torch.sqrt(variance + epsilon)
        outputs = gamma * normalized + beta
        return outputs


    def forward(self, queries, values, block, field,  k, lambda_1, is_training, dropout_rate):
            
        A =  F.relu(self.A_layers[f'att_b{block}_0'](values))
        H = self.H_layers[f'w_b{block}_f{field}'](values)
        
        if self.has_residual: 
            Q_res = F.relu(self.Q_res_layers[f'res_b{block}_f{field}'](queries))

        A_ = torch.cat(torch.split(A, A.size(-1) // self.num_heads, dim=2), dim=0) 
        H_ = torch.cat(torch.split(H, H.size(-1) // self.num_heads, dim=2), dim=0) 
        
        S = F.relu(self.S_layers_1[f'gsl_1_b{block}_0'](values))
        S = torch.sigmoid(self.S_layers_2[f'gsl_2_b{block}_0'](S))
            
        S = S.squeeze(-1) 
        vals, inds = torch.topk(S, k=k, dim=-1)

        kth, _ = torch.min(vals, dim=-1, keepdim=True)
        topk = (S >= kth).float() 
        S = S * topk
        S = S.unsqueeze(-1) 
        vis = S.transpose(1, 2) 
        S = S.repeat(self.num_heads, 1, 1)
        A_ = S * A_
        A_ *= lambda_1

        weights = F.softmax(A_, dim=1)
        weights = F.dropout(weights, p=1 - dropout_rate,  training=self.training)

        outputs = weights * H_ 
        outputs = torch.sum(outputs, dim=1, keepdim=True)

        outputs = torch.cat(torch.split(outputs, outputs.size(0) // self.num_heads, dim=0), dim=2)

        if self.has_residual:
            outputs += Q_res

        outputs = F.relu(outputs)
        outputs = F.layer_norm(outputs, outputs.size()[1:])

        return outputs, vis


class PGNN(nn.Module):
    def __init__(self, opt, feature_size, user_num, item_num, input_dim, user_res, ad_res, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2, layer_num=3):
        super(PGNN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        
        self.user_res = user_res
        self.ad_res = ad_res
        
        self.user_pred_embedding = nn.Embedding(self.user_res.shape[0], self.user_res.shape[1])
        self.user_pred_embedding.weight.data.copy_(self.user_res.cuda())
        self.user_pred_embedding.weight.requires_grad = False
        
        pad_vector = torch.randn(1, self.ad_res.shape[1])
        self.ad_res = torch.cat((self.ad_res, pad_vector), dim=0)
        
        self.item_pred_embedding = nn.Embedding(self.ad_res.shape[0], self.ad_res.shape[1])
        self.item_pred_embedding.weight.data.copy_(self.ad_res.cuda())
        self.item_pred_embedding.weight.requires_grad = False

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)

        self.unified_map = nn.Linear((self.layer_num * 2 + 1) * self.hidden_size, self.hidden_size, bias=False)

            
        self.layers = nn.ModuleList([PGNNLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop) for _ in range(self.layer_num)])
        
#         self.multi_head_attention = AttentionLayer(FullAttention(mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True), self.user_hidden_size, 4)
        
#         self.target_attention = AttentionSequencePoolingLayer(att_hidden_units=(self.user_hidden_size*4, self.user_hidden_size), weight_normalization=True, supports_masking=False, embedding_dim = self.user_hidden_size)
#         self.query_emb_map = nn.Linear(self.hidden_size, self.user_hidden_size, bias=False)
        
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.final_fc = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.feature_size = feature_size
        self.field_size = opt.field_size
        self.embedding_size = opt.embedding_size
        self.blocks = opt.blocks
        self.heads = opt.heads
        self.block_shape = opt.block_shape
        self.has_residual = opt.has_residual
        self.ks = opt.ks

        self.feature_embeddings = nn.Embedding(feature_size, self.embedding_size)
        nn.init.xavier_uniform_(self.feature_embeddings.weight)

        input_size = self.field_size * self.embedding_size

        self.prediction = nn.Linear(input_size, 1)

        self.weights = {}
        self.weights["lambda_2"] = torch.nn.Parameter(torch.normal(mean=0.0, std=1, size=()), requires_grad=True)
        self.weights["lambda_1"] = torch.nn.Parameter(torch.normal(mean=0.0, std=1, size=()), requires_grad=True)   
        
        self.gat_attention = gat_attention(self.embedding_size, self.heads, self.block_shape, self.field_size, self.has_residual)     
        
        self.final_prediction_layer = torch.nn.Linear(sum(self.block_shape), self.embedding_size).cuda()
        self.final_map = torch.nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size).cuda()
        
        self.reset_parameters()

    def forward(self, user_org=None,  user_index=None, g=None,  target_org=None, target_index=None, last_item_index=None, feat_index=None, u_t=None, ad_t=None, dropout_keep_prob=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        g.nodes['user'].data['user_h'][user_org] = self.user_pred_embedding(u_t.cuda()).cuda()
        g.nodes['item'].data['item_h'][target_org] = self.item_pred_embedding(ad_t.cuda()).cuda()

        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
                user_layer.append(graph_item(g, target_index, feat_dict['item']))
                
            item_embed = graph_item(g, last_item_index, feat_dict['item'])
            user_layer.append(item_embed)

        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
    
        embeddings = self.feature_embeddings(feat_index)
        embeddings = F.dropout(embeddings, p=1-dropout_keep_prob[1], training=is_training)
        self.y_deep = embeddings
        h = self.y_deep
        h_list = []
        v_list = []

        for i in range(self.blocks): # 3 the layer of feature interaction
            state_list = []
            vis_list = []
            neighbor_list = []

            for j in range(self.field_size):
                j_row = h[:, j:j+1, :]
                lamb_y_deep = self.weights['lambda_2'] * h
                neighbor = torch.cat([lamb_y_deep[:, :j, :], j_row, lamb_y_deep[:, j+1:, :]], dim=1)
                neighbor = torch.sum(neighbor, dim=1) / (1.0 + self.weights["lambda_2"] * (self.y_deep.shape[1] - 1))
                neighbor_list.append(neighbor)
                neighbor_hat = torch.stack(neighbor_list, dim=1)

            for j in range(self.field_size):
                state = neighbor_hat[:, j:j+1, :]
                state_all = state * neighbor_hat
                state, vis = self.gat_attention(queries=state,
                                                  values=state_all,                                             
                                                  block=i,
                                                  field=j, 
                                                  k=self.ks[i],
                                                  lambda_1=self.weights['lambda_1'], 
                                                  is_training = is_training,
                                                  dropout_rate = dropout_keep_prob[0])
                state_list.append(state)
                vis_list.append(vis)
                
            h = torch.cat(state_list, dim=1)
            v = torch.cat(vis_list, dim=1)
            h_list.append(h)
            v_list.append(v)

        self.v_list = torch.transpose(torch.stack(v_list, dim=0), 0, 1) 
        self.y_deep = torch.cat(h_list, dim=-1)
        self.flat = torch.mean(self.y_deep, dim=1)

        self.out = self.final_prediction_layer(self.flat)#(b, hidden)

        final_embedding = torch.cat((unified_embedding, self.out), dim=-1)

        final_embedding = self.final_map(final_embedding)
        
        score = torch.matmul(final_embedding, self.item_embedding.weight.transpose(1, 0))#Equation(18)
        y = self.final_fc(final_embedding).squeeze(1)

        return score, y#, attn, target_attn

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

class PGNNLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, K=4):
        super(PGNNLayers, self).__init__()
        self.hidden_size = in_feats
        
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length

        self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
       
        self.u_time_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
        self.u_time_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)
        self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True, bidirectional=True)
        self.gru_linear_u = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        
        self.i_time_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
        self.i_time_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        
        self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True, bidirectional=True)
        self.gru_linear_i = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
       
            
#         self.multi_head_attention_layer_user = AttentionLayer(FullAttention(mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True), self.hidden_size, 8)
#         self.multi_head_attention_layer_item = AttentionLayer(FullAttention(mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True), self.hidden_size, 8)
        

    def user_update_function(self, user_now, user_old):
        
        return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))

    def item_update_function(self, item_now, item_old):
     
        return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
       

    def forward(self, g, feat_index, feat_dict=None):
        if feat_dict == None:
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
           
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        

        return f_dict

    def graph_update(self, g):

        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']    
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']#用户指向商品
        return dic

    def item_reduce_func(self, nodes):
        h = []
        #先根据time排序
        if nodes.mailbox['time'].shape[1] >self.user_max_length:
            nodes.mailbox['time'] = nodes.mailbox['time'][:,:self.user_max_length]
            nodes.mailbox['user_h'] = nodes.mailbox['user_h'][:,:self.user_max_length,:]
            nodes.mailbox['item_h'] = nodes.mailbox['item_h'][:,:self.user_max_length,:]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] -order -1
        length = nodes.mailbox['item_h'].shape[0]

        rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
        att_order = torch.argsort(torch.argsort(torch.sort(nodes.mailbox['time'], 1)[0], 1), 1)
        att_re_order = nodes.mailbox['time'].shape[1] -order -1 

        gru_input = nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order]
        _, hidden_u = self.gru_i(gru_input)
        _ = self.gru_linear_i(_)

        e_ij = torch.sum((self.i_time_encoding(att_re_order) + _) * nodes.mailbox['item_h'],
                         dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float()) #(neighbor, neighbor)
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        hidden_u = torch.sum(alpha * (_ + self.i_time_encoding_k(att_re_order)), dim=1)
        h.append(hidden_u)

        last = torch.argmax(nodes.mailbox['time'], 1)

        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        
        h.append(last_em.squeeze(1))
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_i(torch.cat(h,-1))}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        if nodes.mailbox['time'].shape[1] > self.item_max_length:
            nodes.mailbox['time'] = nodes.mailbox['time'][:,:self.item_max_length]
            nodes.mailbox['user_h'] = nodes.mailbox['user_h'][:,:self.item_max_length,:]
            nodes.mailbox['item_h'] = nodes.mailbox['item_h'][:,:self.item_max_length,:]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        re_order = nodes.mailbox['time'].shape[1] - order -1#（node, max_length）
        length = nodes.mailbox['user_h'].shape[0]

        rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
        att_order = torch.argsort(torch.argsort(torch.sort(nodes.mailbox['time'], 1)[0], 1), 1)
        att_re_order = nodes.mailbox['time'].shape[1] -order -1 

        gru_input = nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order]
    
#             seq_enc, attn = self.multi_head_attention_layer_item(gru_input.cuda(), gru_input.cuda(), gru_input.cuda(), attn_mask=None, tau=None, delta=None)
#             seq_enc, attn = self.multi_head_attention_layer_item(seq_enc.cuda(), seq_enc.cuda(), seq_enc.cuda(), attn_mask=None, tau=None, delta=None)
        _, hidden_u = self.gru_u(gru_input) 
        _ = self.gru_linear_u(_)

        e_ij = torch.sum((self.u_time_encoding(att_re_order) + _) * nodes.mailbox['user_h'],
                         dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float()) #(B, neighbor)
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))#(B, neighbor)
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        hidden_i = torch.sum(alpha * (_ + self.u_time_encoding_k(att_re_order)), dim=1)#(neighbor, hidden)
        h.append(hidden_i)

        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)

        h.append(last_em.squeeze(1))
        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h,-1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index#tensor([ 1,  4,  9, 16])
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic

def collate(data):
    graph = []
    user = []
    target = []
    user_l = []
    target_l = []
    last_item = []
    label = []
    time = []
    feature= []
    u_t = []
    i_t = []
    with open ('../../all_datasets/Alimama/user_ad_res/u_t_dict.pkl','rb') as f:
        user_t_dict = pickle.load(f)
    with open ('../../all_datasets/Alimama/user_ad_res/i_t_dict.pkl','rb') as f:
        item_t_dict = pickle.load(f)
    number=0
    for da in data:
        number+=1
        graph.append(da[0][0])
        user.append(da[1]['user'])#'user': torch.tensor([u])
        target.append(da[1]['target'])
        user_l.append(da[1]['u_alis'])
        target_l.append(da[1]['tar_alis'])
        last_item.append(da[1]['last_alis'])
        label.append(da[1]['label'])#click or not 1 or 0
        time.append(da[1]['cur_time'])
        feature.append(da[1]['feature'])
        u_t.append(user_t_dict[str(da[1]['user'].item()) + '_' + str(da[1]['cur_time'].item())])
        try:
            i_t.append(item_t_dict[str(da[1]['target'].item()) + '_' + str(da[1]['cur_time'].item())])
        except:
            i_t.append(528271)#
    
        if (da[1]['user'].numel()==0) or (da[1]['target'].numel()==0) or (da[1]['u_alis'].numel()==0) or (da[1]['tar_alis'].numel()==0) or (da[1]['last_alis'].numel()==0) or (da[1]['label'].numel()==0) or (da[1]['feature'].numel()==0):
            print(da[1]['user'], da[1]['target'], da[1]['u_alis'], da[1]['tar_alis'], da[1]['last_alis'], da[1]['label'])
            break

    feature = np.array([item.tolist() for item in feature])
    return torch.tensor(user).long(), torch.tensor(user_l).long(),  dgl.batch(graph), torch.tensor(target).long(), torch.tensor(target_l).long(), torch.tensor(last_item).long(), torch.tensor(label).long(), torch.tensor(time).long(), torch.tensor(feature).long(), torch.tensor(u_t).long(), torch.tensor(i_t).long()




