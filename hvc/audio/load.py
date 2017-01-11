import numpy as np

def read_cbin(filename):
    """
    loads .cbin files output by EvTAF
    """

    data = np.fromfile(filename,dtype=">d") # ">d" means big endian, double
    return data

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
                line = lime_tmp
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
            elif "T after" in line:
                ind = line.find('=')
                rec_dict['time_after'] = float(line[ind+1:])
            elif "T Before" in line:
                ind = line.find('=')
                rec_dict['time before'] = float(line[ind+1:])
            elif "Output Sound File" in line:
                ind = line.find('=')
                rec_dict['outfile'] = int(line[ind+1:])
            elif "thresholds" in line:
                th_list = []
                while 1:
                    line = recfile.line()
                    if line == "":
                        break
                    try:
                        th_list.append(float)
                    except ValueError:  # because we reached next section
                        line_tmp = line
                        break
                rec_dict['thresholds'] = th_list
                if line = "":
                    break
            elif "feedback information" in line:
                fb_dict = {}
                while 1:
                    line = recfile.readline()
                    if line = "":
                        break
                    elif line = "\n":
                        continue
                    ind = line.find("msec")
                    time = float(line[:ind-1])
                    ind = line.find(":")
                    fb_type = line[ind+2:]
                    fb_dict[time] = fb_type
                rec_dict['feedback_info'] = fb_dict
                if line = "":
                    break
            elif "trigger times" in line:
                pass
            elif "file created" in line:
                pass
    return rec_dict
