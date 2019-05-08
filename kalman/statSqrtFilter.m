function [X,P,rejSamples,logL]=statSqrtFilter(Y,A,C,Q,R,varargin)
%statKalmanFilter implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996
%Fast implementation by assuming that filter's steady-state is reached after 20 steps
%INPUTS:
%
%OUTPUTS:
%
%See also: statKalmanSmoother, statKalmanFilterConstrained, KFupdate, KFpredict

% For the filter to be well-defined, it is necessary that the quantity w=C'*inv(R+C*P*C')*y
% be well defined, for all observations y with some reasonable definition of inv().
% Naturally, this is the case if R+C*P*C' is invertible at each step. In
% turn, this is always the case if R is invertible, as P is positive semidef.
% There may exist situations where w is well defined even if R+C*P*C' is
% not invertible (which implies R is non inv). %This requires both: the
% projection of y and of the columns of C onto the 'uninvertible' subspace
% to be  always 0. In that case the output space can be 'compressed' to a
% smaller dimensional one by eliminating nuisance dimensions. This can be done because
% neither the state projects onto those dims, nor the observations fall in it.
% Such a reduction of the output space can be done for efficiency even if
% the projection of y is non-zero, provided that R is invertible and the
% structure of R decouples those dimensions from the rest (i.e. the
% observations along those dims are uncorrelated to the dims corresponding
% to the span of C). Naturally, this is a very special case, but there are
% some easy-to-test sufficient conditions: if R is diagonal, positive, and
% rank(C)<dim(R), compression is always possible.


[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,varargin);
M=processFastFlag(opts.fastFlag,A,N);
if M~=N && any(isnan(Y(:)))
    warning('statKFfast:NaN','Requested fast filtering but data contains NaNs. No steady-state can be found for filtering. Filtering will not be exact. Proceed at your own risk.')
    %opts.fastFlag=false;
    %allows for fast-filtering anyway. The steady-state of K (but not P) may exist
    %under some special circumstances (e.g. missing data forms a regular
    %pattern, such as every other datapoint), and even if not, fast
    %filtering may still be good enough and thus acceptable. Issue warning
    %to make subject aware.
end

%TODO: Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable

%Size checks:
%TODO

if nargout>6 %This is here to make sure that no code is using the legacy call with 7 outputs
    error('Too many outputs')
end

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(D1,N+1,'gpuArray');      X=nan(D1,N,'gpuArray');
    Pp=nan(D1,D1,N+1,'gpuArray');   P=nan(D1,D1,N,'gpuArray');
    rejSamples=false(D2,N,'gpuArray');
else
    Xp=nan(D1,N+1);      X=nan(D1,N);
    Pp=nan(D1,D1,N+1);   P=nan(D1,D1,N);
    rejSamples=false(D2,N);
end

%Priors:
prevX=x0; prevP=P0; Xp(:,1)=x0; Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Check if any output dimensions have 0 or Infinite variances:
infVar=isinf(diag(R));
if any(infVar) %Any infinite variance is equivalent to ignoring the corresponding variables: it does not matter for the filtering itself.
    %However, it will lead to -Inf log-likelihood (adds a -Inf offset associated with each infinite eigenvalue of R).
    %Thus, I remove the relevant components to get a non-infinite log-likelihood.
    warning('statKF:infObsVar','Provided model has infinite variance for some observation components. Ignoring those components for log-likelihood calculation purposes.');
    Y_D=Y_D(~infVar,:);
    R=R(~infVar,~infVar);
    %D=D(~infVar,:);
    C=C(~infVar,:);
end
[~,dR]=ldl(R);
zeroVar=(diag(dR)==0);
if any(zeroVar) %0 variance is very problematic: it means that the output equation is at least partially deterministic, which doesn't bode well with the stochastic framework
    error('statKF:zeroObsVar','Provided model has 0 observation variance for some dimensions, this is incompatible with the Kalman framework (and unlikely in reality!). If uncertainties are truly 0, try reducing the model by separating the deterministic and stochastic components');
    %To do: offer a function to decouple deterministic and stochastic
    %observation dimensions, and run appropriate state estimation
    %algorithms on each of the components (where naturally the stochastic
    %part will be equivalent to the kalman filter projected onto the states
    %known exactly from the deterministic part)
end

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
rejThreshold=NaN;
if opts.outlierFlag
  rejThreshold=chi2inv(.99,D2);
end

%Reduce model if convenient for efficiency:
if D2>D1 && ~opts.noReduceFlag %Reducing dimension of problem for speed
    [CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D);
    C=CtRinvC; R=CtRinvC; Y_D=CtRinvY;  D2=D1; rejSamples=rejSamples(1:D1,:);
else
    logLmargin=0;
end

%For the first steps do an information update if P0 contains infinite elements
firstInd=1;
infVariances=isinf(diag(prevP));
if any(infVariances) %At least one initial variance was infinite
    %Run info filter until D1 non-nan samples have been processed: this is enough to resolve all uncertainties down from infinity if the system is observable
    %Strictly speaking, we should only need to process N = G - D2 + 1 ;
    %non-NaN samples, where G is the number of infinity uncertainties, D2 is the
    %dimension of the observation. This is true for an observable system,
    %otherwise we would need to find the observable/unobservable
    %decomposition and use the number of dimensions corresponding to the
    %observable part.
    Nsamp=find(cumsum(~any(isnan(Y_D),1))>=D1,1,'first');
    [ii,I,ip,Ip]=statInfoFilter2(Y_D(:,1:Nsamp),A,C,Q,R,prevX,prevP,B,zeros(D2,size(U,1)),U(:,1:Nsamp),opts);
    for kk=1:Nsamp
        logL(kk)=-Inf; %Warning: if variance was inifinte, then logL(firstInd)=-Inf!
        %Transform results from information to state estimates:
        [X(:,kk),P(:,:,kk)]=info2state(ii(:,kk),I(:,:,kk));
        [prevX,prevP]=info2state(ip(:,kk+1),Ip(:,:,kk+1));
    end
    firstInd=Nsamp+1;
end

%Run filter for remaining steps:
M=max([M,firstInd+find(cumsum(~any(isnan(Y_D(:,firstInd+1:end)),1))>=D1,1,'first')]); %Ensures at least 1 non-nan sample is processed
cQt=mycholcov2(Q)';
cPt=mycholcov2(prevP)';
cRt=chol(R)';
At=A';
Ct=C';

%Do just the update for i=firstInd; (equivalent to predict+update, when A=I, Q=0)
i=firstInd;
[prevX,cPt,K,logL(i),rejSamples(i)] = sqrtPredictUpdate(Y_D(:,i),prevX, cPt, cRt, zeros(size(cQt)), eye(size(At)), Ct,zeros(size(prevX)),rejThreshold);
X(:,i)=prevX;  P(:,:,i)=cPt; %Store results

%Do predict+update for remaining steps:
for i=(firstInd+1):M
  y=Y_D(:,i); %Output at this step
  b=BU(:,i-1);
  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     [prevX,cPt,K,logL(i),rejSamples(i)] = sqrtPredictUpdate(y,prevX, cPt, cRt, cQt, At, Ct,b,rejThreshold);
  else %Predict only: achieving by forcibly rejecting the update
      [prevX,cPt] = sqrtPredictUpdate(y,prevX, cPt, cRt, cQt, At, Ct,b,1e-100);
  end
  X(:,i)=prevX;  P(:,:,i)=cPt; %Store results
end

if M<N %Do the fast filtering for any remaining steps:
%(from here on, we assume stady-state behavior to improve speed).
    %Steady-state matrices:
    prevX=X(:,M); Psteady=P(:,:,M); %Steady-state UPDATED state and uncertainty matrix
    Ksteady=K;
    Gsteady=eye(size(prevP))-K*C;% %I-K*C,

    %Pre-compute matrices to reduce computing time:
    GBU_KY=Gsteady*BU(:,M:N-1)+Ksteady*Y_D(:,M+1:N); %The off-ordering is because we are doing predict (which depends on U(:,i-1)) and update (which depends on Y(:,i)
    GA=Gsteady*A;

    %Assign all UPDATED uncertainty matrices:
    P(:,:,M+1:end)=repmat(Psteady,1,1,N-M);

    %Check that outlier flag is disabled and that no data is missing
    if opts.outlierFlag %|| any(isnan(GBU_KY(:)))%Should never happen in fast mode
       error('KFfilter:outlierRejectFast','Outlier rejection is incompatible with fast mode.')
    end

    %Loop for remaining steps to compute x:
    for i=M+1:N
        gbu_ky=GBU_KY(:,i-M);
        if ~isnan(gbu_ky) %Have actual observation (not NaN)
            prevX=GA*prevX+gbu_ky; %Predict+Update, in that order.
        else %Observation is missing or rejected, cannot do update
            %Warning: if this happens, then you shouldn't be using fast
            %mode, as the Kalman filter does not reach a steady-state when
            %data is missing. This is, at best, a heuristic calculation.
            %(the value of Psteady defined above is not really
            %steady-state, and Ksteady may or may not be depending on
            %whether the missing data is missing in a regular pattern).
            %Consistently, the value of Pp defined below cannot apply to
            %all samples.
            prevX=A*prevX+BU(:,i); %Predict only;
        end
        X(:,i)=prevX;
    end
end

%Compute mean log-L over samples and dimensions of the output:
if firstInd~=1
    %warning('statKF:logLnoPrior',['Filter was computed from an improper uniform prior as starting point. Ignoring ' num2str(firstInd-1) ' points for computation of log-likelihood.'])
end
aux=logL+logLmargin;
logL=nansum(aux(firstInd:end)); %Full log-L
%logL=nanmean(aux(firstInd:end))/size(Y,1); %Per-sample, per-dimension of output PROVIDED (not necessarily used)
end
