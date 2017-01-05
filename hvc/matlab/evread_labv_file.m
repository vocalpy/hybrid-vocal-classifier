function [outvect, adfreq] = evread_labv_file(filename, channel, orient)

% [outvect, adfreq] = read_labv_file(filename, channel, orient)
%
% READ_LABV_FILE   Read one channel of data from a labview datafile.
%   This function reads one channel (see below for details) of data from
%   the datafile FILENAME. CHANNEL ranges in value from 0-15. READ_LABV_FILE
%   returns in OUTVECT the data from the channel and, in ADFREQ, the input
%   sample frequency. If the optional ORIENT argument is used, and it is set
%   to 'reverse', or 'r', then the channel retrieved by READ_LABV_FILE is the
%   highest channel minus CHANNEL. If the ORIENT arg is not present or is set 
%   to 'normal' or 'n', then the retrieved channel is simply CHANNEL.
%
%   This function is for use with Observer and IO_Stream output files.

error(nargchk(2,3,nargin));

if ~ischar(filename),
  error('filename must be a string');
end
if (channel < 0),
  error('channel must be an integer >= 0');
end
if (nargin == 2)
  orient = 'normal';
end
if ~ischar(orient),
  error('3rd arg must be a string');
end
if ~(strncmp(orient,'n',1) | strncmp(orient,'r',1))
  error('3rd arg must be either "normal" or "reverse"');
end

% init some vars
DEFAULT_ADFREQ = 32000;   % in samples per chan/sec
adfreq = DEFAULT_ADFREQ;
nscans = 0;               % a scan is a sample from all active chans, so total #
								  % of samples = nscans*nchans
nchans = 0;               % #  of active A/D chans

% Get recfile name.
p = findstr(filename, '.');
base = filename(1:p(length(p)));
recfile = [base 'rec'];

% Check that this is an Observer or IO_Stream recfile
if (exist(recfile,'file'))
    %fid = fopen(recfile, 'rt');
    %line = fgetl(fid);
    %%%%%%if (~ischar(line) | ~strncmp(line,'File created', 12)),
    %%%%%%  error([filename ' is not a Labview datafile.']);
    %%%%%%end
    rd=readrecf(recfile);
    
    adfreq=rd.adfreq;
    nscans = rd.nsamp;
    nchans = rd.nchan;

    %%%%%% Get A/D info from recfile
    %while 1,
    %  line = fgetl(fid);
    %  if strncmp(line, 'ADFREQ =', 8),
    %     adfreq = sscanf(line, 'ADFREQ = %d');
    %	  if ~isnumeric(adfreq),
    %        error(['Attempt to read AD freq. in recfile ' recfile '
    %        failed.']);
    %
    %	  end
    %  elseif strncmp(line, 'Samples =', 9),
    %     nscans = sscanf(line, 'Samples = %d');
    %	  if ~isint(nscans),
    %        error(['Attempt to read Sample # in recfile ' recfile ' failed.']);
    %	  end
    %  elseif strncmp(line, 'Chans =', 7),
    %     nchans = sscanf(line, 'Chans = %d');
    %	  if ~isint(nchans),
    %		 error(['Attempt to read # of channels in recfile ' recfile ' failed.']);
    %     end
    %  elseif ~ischar(line),
    %	  break;
    %  end
    %end
    %fclose(fid);
else
    disp(['No Rec file : ',recfile,' Using Def']);
    adfreq=32e3;
    nchans=2;
    nscans=0;
end

if strncmp(orient,'r',1),
   channel = nchans-channel-1;
end
if (channel > nchans-1 | channel < 0)
   error('Selected channel is out of range.');
end
[fid, msg] = fopen(filename, 'r','b');
if (~isnumeric(fid) | ~isempty(msg))
	error(['Could not open file ' filename]);
end

% skip over first CHANNEL # of data points
[skipdata, count] = fread(fid, channel, 'short');
if (count ~= channel),
   error(['Could not read from datafile ' filename]);
end

% Get data from specified channel only, skip other channels
skip = (nchans-1)*2;
[outvect, count] = fread(fid, inf, 'short', skip);
fclose(fid);%count;
if (count ~= nscans)
   warning('Number of data points read does not match the # of samples in recfile.');
end
