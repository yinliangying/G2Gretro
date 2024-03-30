from retroformer.utils.smiles_utils import *
from retroformer.utils.translate_utils import translate_batch_original, translate_batch_stepwise
from retroformer.utils.build_utils import build_model, build_iterator, load_checkpoint,collate_fn
from retroformer.translate import translate
from functools import partial
import re
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device GPU/CPU')
parser.add_argument('--batch_size_val', type=int, default=4, help='batch size')
parser.add_argument('--batch_size_trn', type=int, default=4, help='batch size')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')
parser.add_argument('--stepwise', type=str, default=False, choices=['True', 'False'], help='')
parser.add_argument('--use_template', type=str, default=False, choices=['True', 'False'], help='')

parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--known_class', type=str, default='True', help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', type=str, default=False, choices=['True', 'False'], help='whether sharing vocab')
parser.add_argument('--shared_encoder', type=str, default=False, choices=['True', 'False'],
                    help='whether sharing encoder')

#parser.add_argument('--data_dir', type=str, default='./data/template', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='./intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
parser.add_argument('--checkpoint', type=str, help='checkpoint model file')

args = parser.parse_args()

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from torch.utils.data import DataLoader
from retroformer.dataset import RetroDataset
from urllib.parse import unquote

# 定义请求处理程序
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # 处理 GET 请求
    def do_GET(self):
        # 解析 URL 和查询参数
        url_parts = urlparse(self.path)
        query_params = parse_qs(url_parts.query)
        print(url_parts)
        print(query_params)
        # 设置响应状态码
        self.send_response(200)
        # 设置响应头部
        self.send_header('Content-type', 'text/html')
        self.end_headers()



        # 构造响应内容
        try:
            http_tmp_data_folder="./data/http_data"
            encoded_value=query_params["encoded_value"]
            json_dict=json.loads(encoded_value[0])
            smiles_list=json_dict["smiles_list"]
            if os.path.exists(http_tmp_data_folder):
                os.system("rm -r %s"%http_tmp_data_folder)
            os.mkdir(http_tmp_data_folder)
            id_list=[]
            rxn_smiles_list=[]
            for id_,smiles in enumerate(smiles_list):
                rxn_smiles="C>>%s"%(smiles)
                id_list.append(id_)
                rxn_smiles_list.append(rxn_smiles)
            df = pd.DataFrame({
                "id": id_list,
                "reactants>reagents>production":rxn_smiles_list
            })
            df.to_csv("%s/raw_test.csv"%(http_tmp_data_folder))

            dataset = RetroDataset(mode='test', data_folder=http_tmp_data_folder,
                                   intermediate_folder=args.intermediate_dir,
                                   known_class=args.known_class == 'True',
                                   shared_vocab=args.shared_vocab == 'True')
            src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
            iterator = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                                   collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
            ground_truths, generations = translate(iterator, model, dataset)

            if len(generations)!=len(smiles_list):
                raise Exception

            result_list=[]
            for r in generations:
                result_list.append(r.tolist())
            message=json.dumps(result_list)
        except:
            self.send_error(404, "error")
            return


        # 发送响应内容
        self.wfile.write(bytes(message, "utf8"))

# 创建 HTTP 服务器并指定请求处理程序
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


# 运行服务器
if __name__ == '__main__':
    print(args)
    if args.known_class == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '_typed'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_untyped'
    if args.use_template == 'True':
        args.stepwise = 'True'

    # Load Checkpoint Model:
    start_data_dir="./data/start_data/"
    tmp_dataset = RetroDataset(mode='test', data_folder=start_data_dir,
                           intermediate_folder=args.intermediate_dir,
                           known_class=args.known_class == 'True',
                           shared_vocab=args.shared_vocab == 'True')

    model = build_model(args, tmp_dataset.src_itos, tmp_dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)


    run()