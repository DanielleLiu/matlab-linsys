function [x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,aux)
%Argument order: D1, N, x0,P0,B,D,U,opts
%outlierRejection,fastFlag,Ub)

%Defaults:
x0=zeros(D1,1); P0=Inf*eye(D1);  B=zeros(D1,0);  D=zeros(1,0);  U=zeros(size(B,2),N); opts=[];
defaultArgs={x0, P0, B, D, U, opts};

%Replace defaults if given and not empty
emptyIdx=cellfun(@isempty,aux);
auxIdx=1:length(aux);
defaultArgs(auxIdx(~emptyIdx))=aux(~emptyIdx); %Replacing defaults with whatever was given
[x0,P0,B,D,U,opts]=defaultArgs{[1:6]};
if iscell(U)
    Nu=cellfun(@(x) size(x,1),U);
    if any(diff(Nu)~=0)
        error('Multiple inputs given but with inconsistent sizes')
    end
    Nu=Nu(1);
else
    Nu=size(U,1);
end
    

if ~isfield(opts,'fastFlag') || isempty(opts.fastFlag)
  opts.fastFlag=0; %No fast mode
end
if ~isfield(opts,'outlierFlag') || isempty(opts.outlierFlag)
  opts.outlierFlag=false;
elseif opts.fastFlag && opts.outlierFlag
    warning('kalmanOptions:fastAndOutlierFlag','Requested fast mode AND outlier rejection, which is not possible. Disabling fast mode.')
    opts.fastFlag=0;
end
if ~isfield(opts,'noReduceFlag') || isempty(opts.noReduceFlag)
  opts.noReduceFlag=false;
end
if ~isfield(opts,'indD') %Leave empty as empty
  opts.indD=1:Nu;
end
if ~isfield(opts,'indB') %Leave empty as empty
  opts.indB=1:Nu;
end
if Nu~=size(D,2)
  if isempty(D)
    D=zeros(1,Nu);
    warning('D was empty but Ud was not. Replacing D with 0')
  else
    error('Incompatible sizes of D, U')
  end
end
if Nu~=size(B,2)
  if isempty(B)
    B=zeros(D1,Nu);
    warning('B was empty but Ud was not. Replacing B with 0')
  else
    error('Incompatible sizes of B, U')
  end
end
if ~isfield(opts,'noLogL') || isempty(opts.noLogL)
  opts.noLogL=false; %Requesting the computation of log-L by default
end
