function [U,S,V,res,mvs] = svdifp(varargin) 
%
%   SVDIFP: Find a few smallest or largest singular values of an M-by-N matrix A
%       with M >= N. If M < N, the transpose of A will be used for computation.
%
%   S = svdifp(A) returns the smallest singular value S of A.
%
%   S = svdifp(A,K) returns the K smallest singular values S of A.
%
%   S = svdifp(A,K,'L') returns the K largest singular values of A.
%
%   [U, S, V] = svdifp(A,...) returns the singular vectors as well
%   If A is M-by-N and K singular values are computed, then U is M-by-K
%   with orthonormal columns, S is K-by-K diagonal, and V is N-by-K with
%   orthonormal columns.
%
%   [U, S, V, res, mvs] = svdifp(A,...) also returns residuals for corresponding singular
%           triplets defined by res = norm([Av - u;A'u-v]) and the total
%           number of matrix-vector multiplications during the process.
%
%   svdifp(A,opt), svdifp(A,K,opt), svdifp(A,K,'L',opt)
%   take the following optional inputs defined by opt:
%     opt.INITIALVEC:
%       a matrix whose i-th column is the i-th initial approximate right singular
%       vector. default: []     
%     opt.COLAMD:
%       set to 0 to diable column approximate minimum degree permutation;
%       default: 1.
%     opt.TOLERANCE:
%       termination tolerance for outer iteration with residual norm([Av - u;A'u-v]);
%       default: 1e-6 * ||A||_1.
%     opt.MAXIT:
%       set the maximum number of (outer) iterations; default: 1000.
%     opt.INNERIT:
%       set a fixed inner iteration to control the memory requirement;
%       default: between 1 and 128 as adaptively determined.
%     opt.USEPRECON:
%       set preconditioning off or completely on:
%           1. set to 0 to disable preconditioning
%           2. set to 1 to enable preconditioning by built in RIF function
%           3. set to a matrix which is the L factor of incomplete cholesky
%           factoization of A'A
%           4. set to a function handle which approximate A'A^{-1}
%        default: 1
%     opt.SHIFT:
%       set shift to  an approximated singular value; default: 0.
%     opt.ADSHIFT: 
%       set adshift to 1 to choose shift adaptively;
%       default: 0 if computing the smallest singular value and 1 if
%       computing the largest singular value.
%     opt.RIFTHRESH:   
%       a threshold between 0 and 1 used in RIF for computing
%           preconditioner; default: 1e-3;
%           If set to 0, the preconditioner is constructed by the QR
%           factorization.
%     opt.ZRIFTHRESH:
%       a threshold between 0 and 1 used for dropping the Z-factor in RIF;
%           default: 1e-8.
%     opt.RIFNNZ:
%       a number between 1 and n which preallocates the nonzeros in
%           each column of Z in RIF. default: n
%     opt.DISP:
%       set to 0 to disable on-screen display of output, and to any other
%       numerical value to enable display; Default: 1.
%   
%  SVDIFP computes the singular values of A through computing the eigenvalues
%  of A'A using the inverse free preconditioned Krylov subspace method
%  (EIGIFP). The basic SVDIFP algorithm is described in
%      
%       Q. Liang, Q. Ye, Computing Singular Values of Large Matrices
%       With Inverse Free Preconditioned Krylov Subspace Method, submitted
%
%  SVDIFP uses preconditioning constructed by the robust incomplete factorization (RIF)
%  (see M. Benzi and M. Tuma. A Robust Preconditioner with Low Memory Requirements for Large
%  Sparse Least Squares Problems, SIAM J. Scientific Computing, 25 (2003), 499-512).
%  Two routines of RIF (RIF.c or RIFLDL.m) are provided with RIF.c being the
%  more efficient choice. To use RIF.c, invoke MATLAB's mex functionality by
%       mex -largeArrayDims RIF.c 
%  Our RIF routines are based on the implementation of rifnrimex.F by Benzi
%  and Tuma available at http://www2.cs.cas.cz/~tuma/sparslab.html.
%
%  SVDIFP is developed by Qiao Liang (qiao.liang@uky.edu) and Qiang Ye
%  (qiang.ye@uky.edu). This work was supported by NSF under Grant DMS-1317424.
%
%  This program is provided for research and educational use and is
%  distributed through http://www.ms.uky.edu/~qye/software.
%  Neither redistribution nor commercial use is permitted without
%  consent of the authors.
%
%  This version dated March 11 2014



% process inputs
A = varargin{1};

if ~issparse(A) 
	A = sparse(A);
	fprintf(1, 'Warning: A is not in sparse format.\n');             
end  

[nr,nc] = size(A);
transflag = 0; % whether the transpose of A is computed
if nr < nc
    A = A';
    transflag= 1;
    [nr,nc] = size(A);
    fprintf(1, 'Warning: m < n, the transpose of A will be computed!\n');
end

% Some default values
k = []; % number of singular values to be computed
maxsym = []; % 'L' if the largest singular values are desired
FindMax = 0; % flag of finding largest singular values
outputYes = 1; % whether output the intermediate results
shift = []; % approximation of the desired singular value
rifthresh = 0.001; % dropping tolerance in L
iterMax = 1000; % number of outer iterations
tolerance = []; % tolerance of outer iteration
V = []; % initial approximations of desired right singular vectors
adaptiveshift = 0; % whether apdaptive shift is enabled
rifnnz = nc; % number of nonzeros in Z of RIF
zrifthresh = 1e-8; % dropping tolerance in Z of RIF
usePrecon = 1; % whether preconditioning is enabled
mValue = 0; % dimension of Krylov subspace
iscolamd = 1; % whether column approximate minimum degree permutation is enabled

% Check Inputs: 
if nargin > 1 
    if ~isstruct(varargin{2})
        k = varargin{2};
        if ~isnumeric(k) || ~isscalar(k) || ~isreal(k) || (k>nc) || ...
                (k<0) || ~isfinite(k)
            error('SVDIFP: Number of singulars requested, k, must be a positive integer <= n');
        end
        if round(k) ~= k
           fprintf(1, 'Warning: Number of singulars requested, k, must be a positive integer <= n. Rounded number used\n'); 
           k = round(k);
        end
    end
    if (nargin > 2) && (~isstruct(varargin{3}))
        maxsym = varargin{3};
        FindMax = 1;
        if ~strcmp(maxsym,'L')
            error('SVDIFP: Third argument must be the string ''L''.')
        end
    end
end

% Get options
if nargin > 1 + ~isempty(k) + ~isempty(maxsym)
    options = varargin{2 + ~isempty(k) + ~isempty(maxsym):nargin};
    if ~isstruct(options)
        error('SVDIFP: Input Error: too many input parameters.'); 
    end
    
    names = fieldnames(options);   
    % Check whether initial approximate right singular vectors are given
    I = strcmpi('INITIALVEC',names);
    if any(I)
        V = options.(names{I});
        if size(V,1) ~= nc
            if size(V,1) == nr
                V = A' * V;
                fprintf(1, 'Warning: m<n, provided initial vector will be changed to be A''*X!\n');
            else
                error('SVDIFP: Input Error: incorrect size of initial vectors');
            end
        end
        if size(V,2) ~= k
           fprintf(1,'Warning: Number of initial vectors not equal to k\n');
        end    
    end
    
    % Check whether COLAMD is disabled
    I = strcmpi('COLAMD',names);
    if any(I)
        iscolamd = options.(names{I});
    end
    
    % Check whether tolerance of svdifp is given
    I = strcmpi('TOL',names);
    if any(I)
       tolerance = options.(names{I});
       if(tolerance <= 0)
           error('SVDIFP: Input Error: Invalid tolerance input.');
       end   
    end

    % Check whether number of maximum outer iterations of svdifp is given
    I = strcmpi('MAXIT',names);
    if any(I)
        iterMax = options.(names{I});
    end
    
    % Check whether  dimension of Krylov subspace is given
    I = strcmpi('INNERIT',names);
    if any(I)
        mValue = options.(names{I});
    end
        
    % Check whether preconditioning is disabled        
    I = strcmpi('USEPRECON',names);
	if any(I)
        usePrecon = options.(names{I});
        if ~isa(usePrecon,'function_handle') && ~isnumeric(usePrecon)
           error('SVDIFP: usePrecon has to be 0,1, a matrix, or a function handle'); 
        end
        if isnumeric(usePrecon) && ~isscalar(usePrecon) && size(usePrecon,1) ~= nc
            error('SVDIFP: dimension of usePrecon has to be the same as the number of columns of A');
        end
	end

    % Check whether approximate singular value is given
    I = strcmpi('SHIFT',names);
    if any(I)
       shift = options.(names{I});
    end
    
	% Check whether shift is determined by adaptive method
    I = strcmpi('ADSHIFT', names);
    if any(I) 
        adaptiveshift = options.(names{I});        
        if ~isempty(shift) && adaptiveshift ~= 0
           shift = [];
           fprintf(1,'Warning: Choosing shift adaptively, shift provided will not be used!\n'); 
        end        
    end  

    % Check whether dropping thershold of L in RIF is given
    I = strcmpi('RIFTHRESH',names);
    if any(I)
        rifthresh = options.(names{I});
    end
    
    % Check whether dropping threshold of Z in RIF is given
    I = strcmpi('ZRIFTHRESH',names);
    if any(I)
        zrifthresh = options.(names{I});
    end
    
    % Check whether rifnnz of RIF is given
    I = strcmpi('RIFNNZ',names);
    if any(I)
        rifnnz = options.(names{I});
    end
    
    % Check whether display option is off
    I = strcmpi('DISP',names);
    if any(I)
       outputYes = options.(names{I});
    end
end

% If COLAMD is disabled, print warning message
PM = speye(nc,nc);
if iscolamd     
    p = colamd(A);
    A = A(:,p);
    PM = PM(:,p);
else 
    fprintf(1, 'Warning: COLAMD is disabled!\n');
end

% If k is not given, set it to 1
if isempty(k)
    k = 1;
end

% If shift is not given and adaptive shift is diabled, set it to 0
if isempty(shift) && adaptiveshift == 0
   shift = 0; 
end

% Get ||A||_1, for tolerance and deflation
normA = norm(A,1);

% if tolerance is not given, set it to default value: 1e-6 * ||A||_1
if isempty(tolerance)
    tolerance = 1e-6 * normA;
end

% Main function
[U, S, V,res, mvs] = ifreesvd(A,normA, FindMax, mValue,shift,rifthresh,zrifthresh,rifnnz,...
    tolerance,iterMax,k,V,outputYes,usePrecon);
V = PM * V;

S = diag(S);
if nargout <= 1
    U = S;
else 
    if transflag
        tmp = U;
        U = V;
        V = tmp;
    end
end
  
   
function [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs] = ...
     parnob(C, FindMax, L, mvs, lambda, v, r, diff, rDiff, m, tol, V_C,  lambda_1, lambda_max)
%  
%   generate orthonormal basis V of preconditioned (by L) Krylov subspace
%   by m steps of Arnoldi algorithm and then let W be the orthonormal
%   basis of C * V, the approximate singular triplet is extracted from
%   W'CV   
%

% initialization
[nr,nc] = size(C);
% deflation for converged singular vectors
if nargin == 14
    count = size(V_C,2);
    if FindMax == 1
        % converged largest singular values are shifted to 0
        K = speye(count);
    end
end

% C * V = W * Bm
V = zeros(nc,m); % basis of Krylov subspace(right search space)
W = zeros(nr,m); % basis of C * V(left search space)
Vr = V; % residual of columns of V as approximated eigenvectors of C' * C
Bm = zeros(m,m); % projected matrix of C

V(:,1) = v; % v should be normalized
Vr(:,1) = r;
W(:,1) = C * V(:,1); 
Bm(1,1) = sqrt(W(:,1)' * W(:,1));
W(:,1) = W(:,1) / Bm(1,1); % W(:,1) is normalized
mvs = mvs + 2;

% Loop for Arnoldi iteration and Gram-Schmidt orthogalization
for i = 2:m-1                
    % Apply preconditioner if given 
    if ~isempty(L)
        % The preconditioner is supposed to be constructed from LDL-factorization of A'A where D is a diagonal matrix of 1   
        if isnumeric(L) 
            r = L \ r;
            r = (L') \ r;    
        else 
            r = feval(L,r); 
        end
    end
    % orthogalize r with V(:,1:i-1) 
    r = mgs(r,V(:,1:i-1)); % Modified Gram Schmidt
%     r = r - V(:,1:i-1) * (V(:,1:i-1)' * r); % Standard Gram Schmidt   
    % reorthogonalization if m > 6 
    if( m > 6)  
        r = mgs(r,V(:,1:i-1)); 
%         r = r - V(:,1:i-1) * (V(:,1:i-1)' * r); 
    end
    
    % deflation if needed
    if nargin == 14
        if FindMax == 1
            r = r - V_C * (K * (V_C' * r));
        end
    end
    
    % break if converged
    tmp = sqrt(r' * r);
    if tmp == 0 
        m = i;
        break 
    end
    
    % normalize and save new basis vector 
    V(:,i) = r / tmp;
    W(:,i) = C * V(:,i);
    r = C' * W(:,i) - lambda * V(:,i);    
    
    % orthogonalize W(:,i)
    [W(:,i),Bm(1:i-1,i)] = mgs(W(:,i),W(:,1:i-1)); % Modified Gram Schmidt 
%     Bm(1:i-1,i) = W(:,1:i-1)' * W(:,i);
%     W(:,i) = W(:,i) - W(:,1:i-1) * Bm(1:i-1,i);
    % reorthogonalization if m > 6 
    if m > 6  
        [W(:,i),tmp] = mgs(W(:,i),W(:,1:i-1)); % Modified Gram Schmidt
%         W(:,i) = W(:,i) - W(:,1:i-1) * (W(:,1:i-1)' * W(:,i));
    end
    Bm(i,i) = sqrt(W(:,i)' * W(:,i));
    W(:,i) = W(:,i) / Bm(i,i);
    Vr(:, i) = r;  
end 

% add the diff vector to the basis
% and complete the projection Bm 
diffNorm = sqrt(diff' * diff);
[diff,tmp] = mgs(diff,V(:,1:m-1));% Modified Gram Schmidt
rDiff = rDiff - Vr(:,1:m-1) * tmp;
% diff = diff - V(:,1:m-1) * (V(:,1:m-1)' * diff);
% rDiff = rDiff - Vr(:,1:m-1) * (V(:,1:m-1)' * diff);

% reorthogonalization if m > 6
if m > 6
    [diff,tmp] = mgs(diff,V(:,1:m-1));% Modified Gram Schmidt
    rDiff = rDiff - Vr(:,1:m-1) * tmp; 
%     diff = diff - V(:,1:m-1) * (V(:,1:m-1)' * diff);
%     rDiff = rDiff - Vr(:,1:m-1) * (V(:,1:m-1)' * diff);
end 

% deflation if needed
if nargin == 14
    if FindMax == 1
        rDiff = rDiff - lambda * diff;
    end
end

tmp = sqrt(diff' * diff);
% check and add diff only if it's significant
if tmp <= 1e-8 * diffNorm
        m = m-1; % diff is insignificant
elseif tmp <= 1e-2 * diffNorm
    % recompute residual of diff i.e. C' * C * diff - sigma ^ 2 * diff if necessary
    V(:,m) = diff / tmp;
    diff = V(:,m); 
    W(:,m) = C * V(:,m);
    rDiff = C' * W(:,m) - lambda * diff;     
    Vr(:,m) = rDiff;    
    
    % orthogonalize W(:,m)
    [W(:,m),Bm(1:m-1,m)] = mgs(W(:,m),W(:,1:m-1)); % Modified Gram Schmidt
%     Bm(1:m-1,m) = W(:,1:m-1)' * W(:,m);
%     W(:,m) = W(:,m) - W(:,1:m-1) * Bm(1:m-1,m);
        
    % reorthogonalization if m > 6 
    if m > 6 
        [W(:,m),tmp] = mgs(W(:,m),W(:,1:m-1));
%         W(:,m) = W(:,m) - W(:,1:m-1) * (W(:,1:m-1)' * W(:,m));
    end
    Bm(m,m) = sqrt(W(:,m)'*W(:,m));    
    W(:,m) = W(:,m) / Bm(m,m);
else     
    V(:,m) = diff / tmp; 
    Vr(:,m) = rDiff / tmp;
    
    % orthogonalize W(:,m)
    W(:,m) = C * V(:,m);
    [W(:,m),Bm(1:m-1,m)] = mgs(W(:,m),W(:,1:m-1)); % Modified Gram Schmidt
%     Bm(1:m-1,m) = W(:,1:m-1)' * W(:,m);
%     W(:,m) = W(:,m) - W(:,1:m-1) * Bm(1:m-1,m);
    % reorthogonalization if m > 6 
    if m > 6
        [W(:,m),tmp] = mgs(W(:,m),W(:,1:m-1)); % Modified Gram Schmidt
%         W(:,m) = W(:,m) - W(:,1:m-1) * (W(:,1:m-1)' * W(:,m));
    end
    Bm(m,m) = sqrt(W(:,m)' * W(:,m));
    W(:,m) = W(:,m) / Bm(m,m);  
end

% number of matrix-vector multiplications
mvs = mvs + 2 * m;

% compute singular triplets of projected matrix

[~,S_B,V_B] = svd(Bm(1:m,1:m)); % S_D are with decreasing order
S_B = diag(S_B);
if FindMax == 0
    svidx = m;
else
    svidx = 1;
end
sigma = S_B(svidx); % approximate singular value
v = V(:,1:m) * V_B(:,svidx); % approximate right singular vector
u = W(:,1:m) * V_B(:, svidx);

% normu and normv are supposed to be one
% renormalization
normv = sqrt(v' * v);
normu = sqrt(u' * u);
v = v / normv; % new approximated of right singular vector
u = u / normu; % new approximated of left singular vector
res = norm([C * v - sigma * u; C' * u - sigma *v]); % residual of new singular triplet

% get the residual of v as eigenvector of C' * C
r = Vr(:,1:m) * V_B(:,svidx) / normv;
tmp = sigma * sigma;
shift = tmp - lambda;
r = r - shift * v;

% get corresponding eigenvalue of C' * C
lambda = tmp;

% update new diff and related vectors for the next iteration   
% V_B(1,svidx) = -(V_B(2:m,svidx)' * V_B(2:m,svidx)) / V_B(1,svidx); 
V_B(1,svidx) = V_B(1,svidx) - 1;
diff = V(:,1:m) * V_B(1:m,svidx);
rDiff = Vr(:,1:m) * V_B(1:m,svidx); 
rDiff = rDiff - shift * diff;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update v and u to minimize [C * v-sigma * u; C' * u - sigma * v]
if sigma > eps
    v0 = v;
    u0 = u;
    v1 = C' * u0;
    v1 = v1 / sqrt(v1' * v1);
    u1 = C * v1;
    u1 = u1 / sqrt(u1' * u1);
    P = [u0 u1;v0 v1];
    AugC = [-sigma * speye(nr) C;C' -sigma * speye(nc)];
    T = P' * (AugC * P);
    [V_T,~] = eig(T);
    tmp = P * V_T(:,1);
    tmpu = tmp(1:nr);
    tmpv = tmp(nr+1:end);
    res = norm([C * tmpv - sigma * tmpu; C' * tmpu - sigma * tmpv]);
else
    res = sqrt(r' * r);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 % recompute r if necessary 
if  m >= 10        
    r = C' * (C * v);
    lambda = (v' * r) / (v' * v); 
    r = r - lambda * v;
    mvs = mvs + 2;
end

% use a new diff if converged. 
if res < tol
    if sigma > eps
        u = tmpu;
        v = tmpv;
    end
    diff = V(:,1:m) * V_B(:,svidx - 1 + 2 * FindMax);
end
 
function [U_C, S_C, V_C,residuals, mvs] = ...
    ifreesvd(C,normC, FindMax, m,shift,rifthresh,zrifthresh,rifnnz, ...
    tol, itermax, k, V,  outputYes,usePrecon)
%
%  The main function that carries out the (outer) iteration. 
%    
%  To be consistent with the notation in the paper, here we use C to denote
%  the target matrix and A = C' * C.
%


[nr,nc] = size(C); % Get size of C

% Compute preconditioner of svdifp
if ~isempty(shift) && isscalar(usePrecon) && usePrecon == 1 && FindMax == 0
    if shift == 0 && rifthresh == 0
        if outputYes ~= 0
            fprintf('===================================================================\n');
            fprintf('Computing precondioner of A using qr, if it takes too long, try increasing opt.rifthresh\n');
            fprintf('====================================================================\n');
        end
        R = qr(C,0);
        L = R';
        startL = 1;
        Lcomputed = 2;
    else
        if outputYes ~= 0
            fprintf('===================================================================\n');
            fprintf('Computing RIF of A, if it takes too long, try increasing opt.rifthresh or decreasing rifnnz\n');
        end
        
        RIFexist = exist('RIF','file');
        if RIFexist == 3
            rifstart = cputime;
            L = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            cputime - rifstart
        else
            fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
            L = RIFLDL(C,shift,rifthresh,zrifthresh);
        end
        if outputYes ~= 0
            fprintf('Done!\n');
            fprintf('====================================================================\n');
        end
        startL = 1;
        Lcomputed = 1;
    end
elseif isscalar(usePrecon) && usePrecon == 0
    startL = 0;
    Lcomputed = 0;
else
    startL = 3;
    L = usePrecon;
    Lcomputed = 1;
end

nnz(L)

% initialization 
S_C = zeros(k,1); % converged k singular values of C
V_C = zeros(nc,k); % converged k right singular vectors of C
U_C = zeros(nr,k); % converged k left singular vectors of C
residuals = zeros(k,1); % residuals of converged k singular triplets
mvs = 0; % number of matrix-vector multiplications

conv = ones(itermax,1); % variables for adaptively choosing m
rate = ones(10,1);

Lambda_1 = zeros(k,1); % converged k eigenvalues of C'C
Lambda_Max = zeros(k,1); % Estimation of largest eigenvalue of C'C

diff = zeros(nc,1); % Initialization of diff vector       
rDiff = diff; % C' * C * diff - sigma ^ 2 * diff


initialConv = 0; % initialization of sign of lth singular triplet converges

% loop for computing k singular triplets
for l = 1:k
    % initialization for each singular value iteration    
   
    % set initial right singular vector if not given
    if size(V,2) >= l        
        v = V(:,l);        
    elseif initialConv == 1 || l == 1
        v = rand(nc,1) - 0.5 * ones(nc,1);
    elseif l > 1 && initialConv == 0 
        v = diff;
    end
    if l > 1
        v = v - V_C(:,1:l) * (V_C(:,1:l)' * v);
    end
    normv = sqrt(v' * v);
    if normv == 0
       error('IFREESVD: Invalid initial left singular vector'); 
    end
    
    v = v / normv; % normalize v
    u = C * v; % initial left singular vector
    lambda = u' * u; % rayleigh quotient of v, approximated eigenvalue of C' * C 
    sigma = sqrt(lambda);% approximate singular value of C, norm of u
    u = u / sigma; % normalize u    
    res = norm([C * v - sigma * u; C' * u - sigma * v]); % residual of singular triplet
    r = C' * (C * v) - lambda * v; % residual of u as eigenvalue of C' * C
    mvs = mvs + 2;
        
    if l > 1
        diff = zeros(nc,1); % Initialization of diff vector       
        rDiff = diff; % C' * C * diff - sigma ^ 2 * diff
    end
    
    mValue = m;
    conv(1) = res;       
    changeCounter = -itermax;
    if mValue < 1
        mValue = min(nc-1,2);
        if startL == 1
            mValue = 1; 
        end
        changeCounter = -10;
    end
    priorM = mValue; 
    priorRate = -eps;
        
    if outputYes ~= 0
        fprintf(1,'\nComputing  %d-th Singular value of A:\n',l);
        fprintf(1,'  Iteration\tSingular Value\tResidual(Av-sigma*u;A''u-sigma*v)\n');
        fprintf(1,'  ======================================================\n');
        fprintf(1,'  %d\t\t%E\t%E\n',1,sigma,res); 
    end
    
    % iteration for computing l-th singular value
    for iter = 2:itermax        
        % if converged, set parameters for next singular value
        if res < tol   
            initialConv = 1;  
            break;
        end
        projSize = mValue + 2;
        % compute Ritz value and vector by Krylov projection   
        if l > 1
          % with deflation for l > 1      
          if startL == 0 
              % without preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff, res, mvs] = ...
              parnob(C, FindMax, [], mvs, lambda, v, r,  diff, rDiff, projSize, tol,...
                          V_C(:,1:l-1),   Lambda_1(1:l-1), Lambda_Max(1:l-1,:));
     
          else
              % with preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs] = ...
              parnob(C, FindMax, L, mvs, lambda, v, r,  diff, rDiff,  projSize, tol,...
                          V_C(:,1:l-1),   Lambda_1(1:l-1), Lambda_Max(1:l-1,:));
                   
          end           
        else
          % no deflation if l = 1. 
          if ( startL == 0 )
              % without preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs] = ...
              parnob(C, FindMax, [], mvs, lambda, v, r,  diff, rDiff,  projSize, tol);
          else
              % with preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs] = ...
              parnob(C, FindMax, L, mvs, lambda, v, r,  diff, rDiff,  projSize, tol);
                   
          end           
        end
        
        if (outputYes ~= 0)
            fprintf(1,'  %d\t\t%E\t%E\n',iter,sigma,res);
        end
         
        
        conv(iter) = res; % residual of each iteration
        % check on convergence rate and update mValue
        changeCounter = changeCounter + 1;
        if changeCounter >= 19 
            rate = [rate(2:10); aveRate(conv, iter - changeCounter + 4, iter)];
            [mValue, priorM, priorRate, fixM] = updateM(rate,mValue,priorM,priorRate,nc);
            changeCounter = (changeCounter + fixM * itermax) * (1 - fixM);  
        elseif changeCounter >= 10 
            rate = [rate(2:10); aveRate(conv, iter - changeCounter + 4, iter)];
        end

        % Check whether compute preconditioner based on convergence behavior
        if (res <= tol * 10 || iter >= itermax / 5) && (startL == 0) && usePrecon == 1 && res > tol
            startL = 1;
            Lcomputed = 1;            
            shift = sigma;
            if outputYes ~= 0
                fprintf('===================================================================\n');
                fprintf('Computing RIF of A using shift %f, if it takes too long, try increaseing opt.rifthresh\n',shift);
            end
            RIFexist = exist('RIF','file');
            if RIFexist == 3
                L = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            else
                fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
                L = RIFLDL(C,shift,rifthresh,zrifthresh);
            end
            if outputYes ~= 0
                fprintf('Done!\n');
                fprintf('====================================================================\n');
                fprintf('Start preconditioned SVDIFP\n');
            end
        end
        
        % check whether compute a new preconditioner based on convergence behavior
        if (((res >= sqrt(tol)) && iter >= itermax/2) || ((res >= tol * 1e2) && iter >=itermax-50))...
                && Lcomputed <= 1 && isscalar(usePrecon) && usePrecon == 1
            Lcomputed = 2;
            shift = sigma;
            if outputYes ~= 0
                fprintf('===================================================================\n');
                fprintf('ReComputing RIF of matrix A using shift %f, if it takes too long, try increasing opt.rifthresh\n',shift);
            end
            if RIFexist == 3
                L = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            else
                fprintf('Couldn''t find RIF.mexa32 or RIF.mexa64, Matlab implementation RIFLDL is used instead\n');
                L = RIFLDL(C,shift,rifthresh,zrifthresh);
            end            
            if outputYes ~= 0
                fprintf('Done!\n');
                fprintf('====================================================================\n');
                fprintf('Start a new preconditioned SVDIFP\n');   
            end
        end
    end

    % store singular triplets
    S_C(l) = sigma;
    V_C(:,l) = v;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % check whether need to recompute left singular vector
    if sigma < eps
        u = rand(nr,1) - 0.5 * ones(nr,1); % initial approximate left singular vector
        normu = sqrt(u' * u);
        u = u / normu;
        tmp = C' * u;
        lambda = tmp' * tmp;
        sigma = sqrt(lambda);
        r = C * (C' * u) - lambda * u;   
        res = sqrt(r' * r);        
        diff = zeros(nr,1); % Initialization of diff vector       
        rDiff = diff; % C' * C * diff - sigma ^ 2 * diff
        
        L = RIFLDL(C',shift,rifthresh,zrifthresh);
        
        if outputYes ~= 0
            fprintf(1,'\nComputing  left Singular vector of A:\n');
            fprintf(1,'  Iteration\tSingular Value\tResidual(A * A'' * u - sigma^2 * u)\n');
            fprintf(1,'  ======================================================\n');
            fprintf(1,'  %d\t\t%E\t%E\n',1,sigma,res); 
        end
        
        for iter = 2:itermax
            [sigma, lambda, u, ~, r,  diff, rDiff, res, mvs] = ...
              parnob(C', 0, L, mvs, 0, u, r,  diff, rDiff, 10, tol);
            if (outputYes ~= 0)
                fprintf(1,'  %d\t\t%E\t%E\n',iter,sigma,res);
            end
            % r = C * C' * u - sigma * sigma * u
            % if r == 0, then v as corresponding eigenvector has converged
            if res < tol
                break;
            end 
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_C(:,l) = u;
    v = V_C(:,l);
    sigma = S_C(l);
    res = norm([C * v - sigma * u; C' * u - sigma * v]);
    residuals(l) = res;
    
    Lambda_1(l) = lambda; % approximations of first l eigenvalues of C' * C
    Lambda_Max(l) = normC; % approximations of biggest eigenvalue of C' * C
    
    % warn if not converged      
    if res >= tol
        fprintf(1,'\n');
        fprintf(1,'Warning: Singular Value %d not converged to the tolerance within max iteration\n',l);
        fprintf(1,'         Residual = %E , the set tolerance = %E \n',res, tol);
        fprintf(1,'         Try decreasing opt.rifthresh, increasing opt.maxit, opt.innerit, or choose a better shift\n');
    end
    if outputYes ~= 0
        fprintf(1,'  ------------\n');
        fprintf(1,'  Singular Value %d converged. \n',l); 
        fprintf(1,'  Residual = %.16E , the set tolerance = %E \n',res, tol);
    end
end

function L = RIFLDL(A,shift,rifthresh,zrifthresh)
%
% Robust Incomplete factorization of A'A
%

[~,n] = size(A); % get number of columns of A
nnzmax = 10000000;
L = spalloc(n,n,nnzmax); % preallocate space for L
threshold = zeros(n,1);
Z = speye(n,n);

for j = 1:n
   tmp = sum(abs(A(:,j)));
   threshold(j) = rifthresh * tmp; % dropping threshold in L
end

for j = 1:n
    % Compute <Azj,Azj> - shift ^ 2 * <zj,zj>    
    % The first column of PrevZ is current zj
    tmpZj = Z(:,1);
    tmpZjnorm = sqrt(tmpZj' * tmpZj);
    tmpZj(abs(tmpZj) < zrifthresh * tmpZjnorm) = 0; % dropping in Zj, change to full format to increase speed?    
    AZj = A * tmpZj; 
    pjj = AZj' * AZj;
    if shift
        pjj = pjj - shift * shift * Z(:,1)' * Z(:,1);
    end    
    L(j,j) = sqrt(abs(pjj));
    if L(j,j) < eps
        if threshold(j) > 0
            L(j,j) = threshold(j);
        else
            L(j,j) = eps;
        end
        Z = Z(:,2:end);
        continue;
    end        
    % Get entries in L
    tmpcol = A(:,j+1:n)' * AZj;
    tmpcol = tmpcol / L(j,j);
    tmpidx = (abs(tmpcol) < threshold(j+1:n));
    tmpcol(tmpidx) = 0; % dropping in L
    L(j+1:n,j) = tmpcol;
    tmpcoef = tmpcol / L(j,j);    
    % Update columns of Z   
    Z = Z(:,2:end) - Z(:,1) * tmpcoef';
end

function [v,q] = mgs(v, U)
%
% Modified Gram-Schimidt process
% return v = (I - UU')v and q = U' * v
[~,n] = size(U);
q = zeros(n,1);
for i = 1:n
    q(i,1) = U(:,i)' * v;
    v = v - q(i,1) * U(:,i);
end

function rate = aveRate(conv, k, iter)
%
% compute average linear convergence rate of conv over steps k+1 to iter 
%
rate = 0; 
if iter-k < 2 || k<1  
    return;
end

y = log10(conv((k+1):iter));
xAve = (iter-k+1) / 2; 
xyAve = ((1:iter-k)*y) / (iter-k)-xAve*sum(y) / (iter-k); 
xAve = ((iter-k) ^ 2 - 1) / 12;
rate = xyAve / xAve;

function [mValue, priorM, priorRate, fixM] = updateM(rate, mValue, priorM, priorRate,n)       
%
%  Adaptive update of mValue: inner iteration 
%
fixM = 0;
maxm = min(n-1, 128);

 % update m when rate stagnates  
if (max(rate)-min(rate)) < 0.1*(-rate(10)) || min(rate) > 0 
    k=2;                
    % increase m by k times, 
     % use larger k if slower convergence 
	if (rate(10) > -0.001) && 8 * mValue <= maxm
        k = 8; 
	elseif rate(10) > -0.01 && 4 * mValue <= maxm
        k = 4;
	end
	% increase m by testing acceleration rate 
	incFlag = (rate(10) / priorRate) * (priorM / mValue); 
    if incFlag > 1.05 
        if 2*mValue > maxm 
            fixM = -1;
        else
            priorM = mValue; 
            priorRate = rate(10); 
            mValue = k * mValue;
            fixM = 1;
        end  
	elseif rate(10) > -0.001 && 2*mValue <= maxm
        mValue = k * mValue;
        fixM = 1;  
	elseif rate(10) > -0.01 && 2*mValue <= maxm
        mValue = k * mValue;
        fixM = 1;   
	elseif incFlag < 0.9
        mValue = priorM;
        fixM = -1;  
    end
end