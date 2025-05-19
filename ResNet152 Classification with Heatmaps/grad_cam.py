import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer    #which layer to use
        self.gradients = None
        self.activations = None
        self.hook_handles = []    #list for hooks 
        self._register_hooks()  

    #doing hook method to get gradients and activations
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        #gradient and activation "hooking" to the target layer
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        #getting tensor from input image
        output, _ = self.model(input_tensor)

        #getting label from the tensor image
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        #getting gradients and features
        self.model.zero_grad()
        class_loss = output[:, target_class].sum()
        class_loss.backward()

        gradients = self.gradients
        activations = self.activations

        #finding average of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        #weighting the activations,summing them up, and normalizing after relu
        heatmap = activations.mean(dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

    #removing hooks for mem
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()