import torch
import torch.distributed as dist


def uniformity_loss_square(features):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T 
    loss = sim.pow(2).mean()
    return loss

def centering_matrix(m):
    J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
    return J_m

def renyi_entropy(matrix, alpha):
    """
    Calculate the alpha-order Renyi entropy of a matrix.
    
    Args:
    - matrix (torch.Tensor): A square matrix (n x n).
    - alpha (float): The order of Renyi entropy.
    
    Returns:
    - torch.Tensor: The Renyi entropy value.
    """
    # Ensure the matrix is square and diagonal elements are 1
    assert matrix.shape[0] == matrix.shape[1] and torch.allclose(matrix.diagonal(), torch.ones(matrix.shape[0]))
    n = matrix.shape[0]
    
    # Calculate the Renyi entropy
    entropy = 1 / alpha * torch.log(torch.trace(torch.pow(matrix/n, alpha)))
    return entropy

def matrix_mutual_information(Z1, Z2, alpha):
    matrix1 = Z1.T @ Z1
    matrix2 = Z2.T @ Z2
    # Calculate individual entropies
    h1 = renyi_entropy(matrix1, alpha)
    h2 = renyi_entropy(matrix2, alpha)
    
    # Calculate the Hadamard product of the two matrices
    hadamard_product = torch.mul(matrix1, matrix2)
    
    # Calculate the joint Renyi entropy
    joint_entropy = renyi_entropy(hadamard_product, alpha)
    
    # Calculate mutual information
    mutual_info = h1 + h2 - joint_entropy
    return mutual_info

def uniformity_loss_TCR(features, uniformity_mu=1., centering=False):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    if centering:
        J_m = centering_matrix(features.shape[0]).detach().to(features.device)
        sim = features.T @ J_m @ features
    else:
        sim = features.T @ features 

    # loss = $- \log \det (\mathbf{I} + mu / m * Z Z^{\top})$
    loss = -torch.logdet(torch.eye(sim.shape[0]).to(features.device) + uniformity_mu / sim.shape[0] * sim)
    return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out