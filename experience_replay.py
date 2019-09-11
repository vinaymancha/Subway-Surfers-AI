# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque
import pyautogui
import pytesseract
from skimage.transform import resize
from pynput import keyboard
import time
import torch
from torch.autograd import Variable



def on_press(key):
    global r_
    global update_r
    global pause
    update_r = False
    if (key==keyboard.Key.alt_gr):        
        r_ = 0
        update_r = True
    elif(key==keyboard.Key.ctrl_r):
        r_ = 20
        update_r = True
    elif(key==keyboard.Key.space):
        r_ = -15
        update_r = True
    elif(key==keyboard.Key.esc):
        pause = True               ## To pause training
        
        
        
key_listener = keyboard.Listener(on_release=on_press)
key_listener.start()


# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])


grayscale = True
height = 128
width = 128

# Making the AI progress on several (n_step) steps
def preprocess(img):
        img_size = (height, width)
        img = resize(np.array(img), img_size)
        if grayscale:
            img = img.mean(-1, keepdims = True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img

def do_action(action):
    if(action == 0):
        pyautogui.mouseDown(x=390, y=318, button='left')
        pyautogui.mouseUp(x=140 , y=318)
        pyautogui.moveTo(x=390, y=318)
    elif (action == 1):
        pyautogui.mouseDown(x=390, y=318, button='left')
        pyautogui.mouseUp(x=640 , y=318)
        pyautogui.moveTo(x=390, y=318)
    elif (action == 2):
        pyautogui.mouseDown(x=390, y=318, button='left')
        pyautogui.mouseUp(x=390 , y=68)
        pyautogui.moveTo(x=390, y=318)
    elif (action == 3):
        pyautogui.mouseDown(x=390, y=318, button='left')
        pyautogui.mouseUp(x=390 , y=568)
        pyautogui.moveTo(x=390, y=318)
    elif(action == 4):
        time.sleep(0.3)    #do nothing
    

class NStepProgress:
    
    def __init__(self, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.n_step = n_step
    
    def __iter__(self):
        pyautogui.click(x=498, y=575, clicks=1, button='left')  #resumes
        pyautogui.moveTo(x=390, y=318)
        state = preprocess(pyautogui.screenshot(region =(0,0,800,630)))    #Screenshot of game
        
        history = deque()
        reward = 0.0
        global r_
        global update_r
        global pause
        pause = False
        update_r = False
        r = 0
        r_ = 0
        la_actions=[]
        la_states=[]
        is_done = True
        
        while True:
            if is_done:
                cx = Variable(torch.zeros(1,256))
                hx = Variable(torch.zeros(1,256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
            
            action, (hx, cx) = self.ai(Variable(torch.from_numpy(np.array([state], dtype = np.float32))), (hx, cx))
            action = action[0][0]
            print(action)
            la_actions.append(action)
            la_states.append(state)
            if len(la_actions) > 3:
                del la_actions[0]
                del la_states[0]
                
            if pause:
                time.sleep(5)
                pause = False
                
            #action part
            do_action(action)
            text = pytesseract.image_to_string(pyautogui.screenshot(region = (398,139,137,35)))   #recognizing text to end game 
            
            if "Score" in text:
                is_done=True
                if len(la_actions)>=3:
                    action = la_actions[-3]
                    print("End "+str(action))
                    state = la_states[-3]
                    history.pop()
                    reward -= 20
                r=-30
            else:
                is_done = False
                r=10
            
            if update_r:
                if len(la_actions)>=3:
                    action_ = la_actions[-2]
                    state_ = la_states[-2]
                    history.pop()
                    history.append(Step(state = state_, action = action_, reward = r_, done = is_done, lstm = (hx, cx)))
                    reward += r_ - 10
                    print("update"+str(action_))
                update_r = False
            
            
            if(action==4):
                r=7
                
                
            next_state = preprocess(pyautogui.screenshot(region =(0,0,800,630)))
            
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done, lstm = (hx, cx)))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                pyautogui.click(x=498, y=575, clicks=1, button='left')  #resumes the game
                pyautogui.moveTo(x=390, y=318)
                state = preprocess(pyautogui.screenshot(region =(0,0,800,630)))
                la_actions=[]
                la_states=[]
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 5000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter) 
            self.buffer.append(entry) 
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity
            self.buffer.popleft()
