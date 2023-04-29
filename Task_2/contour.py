import numpy as np
from scipy.interpolate import RectBivariateSpline

from skimage.filters import sobel
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type
#############################################################################################################
def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   max_px_move=1.0,
                   max_num_iter=2500, convergence=0.1,
                   *,
                   boundary_condition='periodic'):
    
    max_num_iter = int(max_num_iter)
    if max_num_iter <= 0:
        raise ValueError("max_num_iter should be >0.")
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    if boundary_condition not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: "+", ".join(valid_bcs)+'.')
    img = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    img = img.astype(float_dtype, copy=False)
    RGB = img.ndim == 3
    if w_edge != 0:
        if RGB:
            edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                    sobel(img[:, :, 2])]
        else:
            edge = [sobel(img)]
    else:
        edge = [0]
    if RGB:
        img = w_line*np.sum(img, axis=2) \
            + w_edge*sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    snake_xy = snake[:, ::-1]
    x = snake_xy[:, 0].astype(float_dtype)
    y = snake_xy[:, 1].astype(float_dtype)
    n = len(x)
    xsave = np.empty((convergence_order, n), dtype=float_dtype)
    ysave = np.empty((convergence_order, n), dtype=float_dtype)
    eye_n = np.eye(n, dtype=float)
    a = (np.roll(eye_n, -1, axis=0)
         + np.roll(eye_n, -1, axis=1)
         - 2 * eye_n)  
    b = (np.roll(eye_n, -2, axis=0)
         + np.roll(eye_n, -2, axis=1)
         - 4 * np.roll(eye_n, -1, axis=0)
         - 4 * np.roll(eye_n, -1, axis=1)
         + 6 * eye_n) 
    A = -alpha * a + beta * b
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True
    inv = np.linalg.inv(A + gamma * eye_n)
    inv = inv.astype(float_dtype, copy=False)
    for i in range(max_num_iter):
        fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
        fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)

        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :])
                                 + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break

    return np.stack([y, x], axis=1)
#############################################################################################################

def calculate_chain_code(snake):
    chain_code = []
    for i in range(len(snake)):
        x1, y1 = snake[i-1]
        x2, y2 = snake[i]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            if dy > 0:
                code = 0
            else:
                code = 4
        elif dx > 0:
            if dy == 0:
                code = 2
            elif dy > 0:
                code = 1
            else:
                code = 7
        else:
            if dy == 0:
                code = 6
            elif dy > 0:
                code = 5
            else:
                code = 3
        chain_code.append(code)
    return chain_code
#############################################################################################################



