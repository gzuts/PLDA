import torch
import torch.nn.functional as F


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = -((final_outputs).sum(dim=1)).mean()
    return average_loss
