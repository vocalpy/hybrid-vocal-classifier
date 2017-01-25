import numpy as np

def readrecf(filename):
    """
    reads .rec files output by EvTAF
    """

    rec_dict = {}
    with open(filename,'r') as recfile:
        line_tmp = ""
        while 1:
            if line_tmp == "":
                line = recfile.readline()
            else:
                line = line_tmp
                line_tmp = ""
                
            if line == "":  # if End Of File
                break
            elif line == "\n": # if blank line
                continue
            elif "Catch" in line:
                ind = line.find('=')
                rec_dict['iscatch'] = line[ind+1:]
            elif "Chans" in line:
                ind = line.find('=')
                rec_dict['num_channels'] = int(line[ind+1:])
            elif "ADFREQ" in line:
                ind = line.find('=')
                rec_dict['sample_freq'] = int(line[ind+1:])
            elif "Samples" in line:
                ind = line.find('=')
                rec_dict['num_samples'] = int(line[ind+1:])
            elif "T After" in line:
                ind = line.find('=')
                rec_dict['time_after'] = float(line[ind+1:])
            elif "T Before" in line:
                ind = line.find('=')
                rec_dict['time before'] = float(line[ind+1:])
            elif "Output Sound File" in line:
                ind = line.find('=')
                rec_dict['outfile'] = line[ind+1:]
            elif "Thresholds" in line:
                th_list = []
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    try:
                        th_list.append(float(line))
                    except ValueError:  # because we reached next section
                        line_tmp = line
                        break
                rec_dict['thresholds'] = th_list
                if line == "":
                    break
            elif "Feedback information" in line:
                fb_dict = {}
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    elif line == "\n":
                        continue
                    ind = line.find("msec")
                    time = float(line[:ind-1])
                    ind = line.find(":")
                    fb_type = line[ind+2:]
                    fb_dict[time] = fb_type
                rec_dict['feedback_info'] = fb_dict
                if line == "":
                    break
            elif "File created" in line:
                header = [line]
                for counter in range(4):
                    line = recfile.readline()
                    header.append(line)
                rec_dict['header']=header
    return rec_dict

def read_cbin(filename,channel=0):
    """
    loads .cbin files output by EvTAF. 
    
    arguments
    ---------
    filename : string

    channel : integer
        default is 0

    returns
    -------
    data : numpy array
        1-d vector of 16-bit signed integers

    sample_freq : integer
        sampling frequency in Hz. Typically 32000.
    """
    
    # .cbin files are big endian, 16 bit signed int, hence dtype=">i2" below
    data = np.fromfile(filename,dtype=">i2")
    recfile = filename[:-5] + '.rec'
    rec_dict = readrecf(recfile)
    data = data[channel::rec_dict['num_channels']]  # step by number of channels
    sample_freq = rec_dict['sample_freq']
    return data, sample_freq
