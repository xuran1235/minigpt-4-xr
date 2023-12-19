import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    # if gr_img is None:
    #     return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
    # return None ,None,None,chat_state, img_list
def gradio_ask(user_message, chatbot, chat_state):
    # if len(user_message) == 0:
    #     return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

def anno(img_path,prompt,txt_path):
    chatbot = gr.Chatbot(label='MiniGPT-4')
    image=img_path
    img_list = gr.State()
    chat_state=gr.State()
    text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

    image, text_input, upload_button, chat_state, img_list=upload_img(image, text_input, chat_state)
    num_beams=5 # 1 to 10
    temperature=2.0 # 0.1 to 2.0
    text_input1, chatbot, chat_state=gradio_ask(prompt, [], chat_state)
    chatbot, chat_state, img_list=gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    with open (txt_path + "/minigpt_anno.txt", 'w') as f :
        f.write(chatbot[-1][1])
    print(chatbot[-1][1])
    gradio_reset(chat_state, img_list)

root_dir = "/mnt/data1/zy/RLBench_data/Test_AllVariations_011701/"
task_name = "close_box100243"
eposides_path = root_dir + task_name + "/all_variations/episodes/episode" 
# prompt = "The picture shows a robotic arm with gripper, and the door of the microwave oven on the table has been opened. Please describe in deatail how the robotic arm closes the the door of microwave. And the answer should focus on the specific operation of the robotic arm, such as how to rotate, when to close the gripper, and when to release the gripper etc. It is also required that the answer must only be listed in points such as 1.2.3.4 etc."
# prompt = "The picture shows a robotic arm with gripper, and the lid of the laptop on the table has been opened. Please describe in deatail how the robotic arm closes the the lid of laptop. And the answer should focus on the specific operation of the robotic arm, such as how to rotate, when to close the gripper, and when to release the gripper etc. It is also required that the answer must only be listed in points such as 1.2.3.4 etc."
prompt = "The picture shows a robotic arm with gripper, and the lid of the box on the table has been opened. Please describe in deatail how the robotic arm closes the the lid of box. And the answer should focus on the specific operation of the robotic arm, such as how to rotate, when to close the gripper, and when to release the gripper etc. It is also required that the answer must only be listed in points such as 1.2.3.4 etc."
for i in range (0,2):
    eposide_path = eposides_path + str(i)
    anno(eposide_path + "/front_rgb/7.png", prompt, eposide_path)