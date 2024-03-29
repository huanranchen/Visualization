from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
from .. import FormatConverter


class VisualBoard:
    def __init__(self, optimizer=None, name=None, start_iter=0):
        # subprocess.Popen(['tensorboard', '--logdir=runs'])
        time_str = time.strftime("%m-%d-%H%M%S")
        if name is not None:
            self.writer = SummaryWriter(f'runs/{time_str}_{name}')
        else:
            self.writer = SummaryWriter(f'runs/{time_str}')

        self.iter = start_iter
        self.optimizer = optimizer

    def __call__(self, epoch, iter):
        self.iter = iter
        self.writer.add_scalar('misc/epoch', epoch, self.iter)
        if self.optimizer:
            self.writer.add_scalar('misc/learning_rate', self.optimizer.param_groups[0]["lr"], self.iter)

    def write_scalar(self, scalar, name):
        try:
            scalar = scalar.detach().cpu().numpy()
        except:
            scalar = scalar
        self.writer.add_scalar(name, scalar, self.iter)

    def write_tensor(self, im, name):
        self.writer.add_image('attack/'+name, im.detach().cpu(), self.iter)

    def write_cv2(self, im, name):
        im = FormatConverter.bgr_numpy2tensor(im)[0]
        self.writer.add_image(f'attack/{name}', im, self.iter)

    def write_ep_loss(self, ep_loss):
        self.writer.add_scalar('total_loss/ep_loss', ep_loss.detach().cpu().numpy(), self.iter)

    def write_loss(self, loss, det_loss, tv_loss):
        self.writer.add_scalar('total_loss/iter_loss', loss.detach().cpu().numpy(), self.iter)
        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), self.iter)
        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), self.iter)


