import numpy as np
def load_pfm(fname, crop):
    if crop:
        if not os.path.isfile(fname + '.H.pfm'):
            x, scale = load_pfm(fname, False)
            x_ = np.zeros((384, 768), dtype=np.float32)
            for i in range(77, 461):
                for j in range(96, 864):
                    x_[i - 77, j - 96] = x[i, j]
            save_pfm(fname + '.H.pfm', x_, scale)
            return x_, scale
        else:
            fname += '.H.pfm'
    color = None
    width = None
    height = None
    scale = None
    endian = None

    file = open(fname)
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale

load_pfm=load_pfm("/media/patrick/D6FA2E85FA2E624B/Users/Patrick/Documents/_PhD/datasets/scene_flow/Sampler/Monkaa/disparity/0048.pfm")
