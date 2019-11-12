conv_defs = [
        Conv(kernel=[3, 3], stride=2, channel=32),              # first block, input 224x224x3
        ANTBlock(up_sample=1, channel=16, stride=1, repeat=1, group=1, ratio=8),  # second block, input : 112x112x32
        ANTBlock(up_sample=6, channel=24, stride=2, repeat=2, group=2, ratio=8),  # third block, input: 112x112x16
        ANTBlock(up_sample=6, channel=32, stride=2, repeat=3, group=2, ratio=12),  # fourth block, input: 56x56x24
        ANTBlock(up_sample=6, channel=64, stride=2, repeat=4, group=2, ratio=16),  # fifth block, input: 28x28x32
        ANTBlock(up_sample=6, channel=96, stride=1, repeat=3, group=2, ratio=24),  # sixth block, input: 28x28x64
        ANTBlock(up_sample=6, channel=160, stride=2, repeat=3, group=2, ratio=32),  # seventh block, input: 14x14x96
        ANTBlock(up_sample=6, channel=320, stride=1, repeat=1, group=2, ratio=64),  # eighth block, input: 7x7x160
        Conv(kernel=[1, 1], stride=1, channel=1280)
]


def inference_antnet(net, input_layer):
    # NCHW
    net.use_batch_norm = True
    x = net.input_layer(input_layer)    
    
    for i, conv_def in enumerate(conv_defs):
        if isinstance(conv_def, Conv):
            x = net.conv(x, conv_def.channel, (3,3), filter_strides=(2,2), activation='')

        elif isinstance(conv_def, ANTBlock):
            stride = (conv_def.stride, conv_def.stride)

            for j in range(conv_def.repeat):
                prev_output = x
                kernel_size = conv_def.up_sample*x.get_shape().as_list()[1]
                x = net.conv(x, conv_def.channel, (kernel_size, kernel_size), filter_strides=(1,1))


    return x

