function [U,S,V,res,mvs,precs] = svdifp(varargin) 
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
%   [U, S, V] = svdifp(A,...) returns the singular vectors as well with 
%       S a K-by-K diagonal matrix containing K singular values sought, 
%       U an M-by-K matrix containing K left singular vectors,  
%       and V an N-by-K matrix containing K right singular vectors.
%
%   [U, S, V, res] = svdifp(A,...) also returns residuals for each singular
%       triplet defined by 
%         res(i) = norm([A'*A*V(:,i) - S(i)^2*V(:,i)]) 
%       for i=1:K. 
%   [U, S, V, res, mvs] = svdifp(A,...) also returns the total number of 
%       matrix-vector multiplications during the process in mvs.
%
%   svdifp(A,opt), svdifp(A,K,opt), and svdifp(A,K,'L',opt)
%   take the following optional inputs defined by opt:
%     opt.INITIALVEC:
%       a matrix whose i-th column is the i-th initial approximate right singular
%       vector. default: []     
%     opt.COLAMD:
%       set to 0 to diable column approximate minimum degree permutation;
%       default: 1 if using preconditioning and 0 if not using preconditioning.
%     opt.TRIPLETRES:
%       set to 1 to use singular value and singular vectors reiplet residuals defined by
%           res(i) = norm([A*V(:,i) - S(i)*U(:,i);A'*U(:,i) - S(i)*V(:,i)]) 
%       for i=1:K. default: 0
%     opt.TOLERANCE:
%       termination tolerance for residual res(i);
%       default: 1e-6 * norm(A,1).
%     opt.MAXIT:
%       set the maximum number of (outer) iterations; default: 1000.
%     opt.INNERIT:
%       set a fixed inner iteration to control the memory requirement;
%       default: between 1 and 128 as adaptively determined.
%     opt.USEPRECON:
%       set preconditioning off or completely on:
%           1. set to 0 to disable preconditioning
%           2. set to 1 to enable preconditioning by the built-in RIF function
%           3. set to a matrix which is the L factor of incomplete L*D*L'
%           factoization of A'*A - sigma^2 I, where sigma is a target
%           singular value; or set to a function handle which approximates (L*L')^{-1}
%        default: 1
%     opt.SHIFT:
%       set to an approximate singular value; default: 0.
%     opt.ADSHIFT: 
%       set ADSHIFT to 0 to disable adaptive choice of shift; and to any other
%       numerical value to enable it. 
%       default: 0 if computing the smallest singular value and 1 if
%          computing the largest singular value.
%     opt.RIFTHRESH:   
%       a threshold between 0 and 1 used in RIF for computing preconditioner; 
%       default: 1e-3;
%       If set to 0, the preconditioner is constructed by the QR factorization.
%     opt.ZRIFTHRESH:
%       a threshold between 0 and 1 used for dropping the Z-factor in RIF;
%       default: 1e-8.
%     opt.RIFNNZ:
%       a number between 1 and n which preallocates the nonzeros in
%       each column of Z in RIF. default: n
%     opt.DISP:
%       set to 0 to disable on-screen display of output, and to any other
%       numerical value to enable display; Default: 1.
%   
%  SVDIFP computes the singular values of A through computing the eigenvalues
%  of A'A using the inverse free preconditioned Krylov subspace method
%  (EIGIFP). The basic SVDIFP algorithm is described in
%      
%       Q. Liang and Q. Ye, Computing Singular Values of Large Matrices
%       With Inverse Free Preconditioned Krylov Subspace Method. 
%
%  SVDIFP uses preconditioner constructed by the robust incomplete factorization (RIF)
%  (see M. Benzi and M. Tuma. A Robust Preconditioner with Low Memory Requirements for Large
%  Sparse Least Squares Problems, SIAM J. Scientific Computing, 25 (2003), 499-512).
%  Two routines of RIF (RIF.c or RIFLDL.m) are provided with RIF.c being the
%  more efficient choice. To use RIF.c, invoke MATLAB's mex functionality by
%       mex -largeArrayDims RIF.c 
%  Our RIF routines are adapted from the implementation of rifnrimex.F by Benzi
%  and Tuma available at http://www2.cs.cas.cz/~tuma/sparslab.html.
%
%  SVDIFP is developed by Qiao Liang (qiao.liang@uky.edu) and Qiang Ye
%  (qiang.ye@uky.edu). This work was supported by NSF under Grant DMS-1317424.
%
%  This program is provided for research and educational use and is
%  distributed through http://www.ms.uky.edu/~qye/software and GOOGLE Code.
%  Neither redistribution nor commercial use is permitted without
%  consent of the authors.
%
%  Version 1.0, dated May 5 2014

% ------ Main Program --------
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
tripletres = 0;
tolerance = []; % tolerance of outer iteration
V = []; % initial approximations of desired right singular vectors
adaptiveshift = 0; % whether apdaptive shift is enabled
iterMax = 1000; % number of outer iterations
rifnnz = nc; % number of nonzeros in Z of RIF
zrifthresh = 1e-8; % dropping tolerance in Z of RIF
usePrecon = 1; % whether preconditioning is enabled
mValue = 10; % dimension of Krylov subspace
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
    
    % Check whether triplet residual is enabled
    I = strcmpi('TRIPLETRES', names);
    if any(I)
       tripletres = options.(names{I}); 
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


% If precondioner is provided, COLAMD should be disabled
if (isnumeric(usePrecon) && ~isscalar(usePrecon)) || (isscalar(usePrecon) && usePrecon == 0)
    iscolamd = 0;
end

% If COLAMD is disabled, print warning message
PM = speye(nc,nc);
if iscolamd     
    p = colamd(A);
    A = A(:,p);
    if ~isempty(V)
        V = V(p,:);
    end
    PM = PM(:,p);
end

% If k is not given, set it to 1
if isempty(k)
    k = 1;
end

% If shift is not given and adaptive shift is disabled, set it to 0
if isempty(shift) && adaptiveshift == 0
   shift = 0; 
end

% if tolerance is not given, set it to default value: 1e-6 * ||A||_1
if isempty(tolerance)
    normA = norm(A,1);
    tolerance = 1e-6 * normA;
end

% Main function
[U, S, V, res, mvs,precs] = ifreesvd(A, FindMax, mValue,shift,rifthresh,zrifthresh,rifnnz,...
    tolerance,iterMax,k,V,outputYes,usePrecon,tripletres);
V = PM * V;

if nargout <= 1
    U = S;
else 
    if transflag
        tmp = U;
        U = V;
        V = tmp;
    end
end
  
   
function [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs,precs] = ...
     parnob(C, FindMax, L, mvs,precs, lambda, v, r, diff, rDiff, m, tol,tripletres, V_C)
%  
%   generate orthonormal basis V of preconditioned (by L) Krylov subspace
%   by m steps of Arnoldi algorithm and then let W be the orthonormal
%   basis of C * V, the approximate singular triplet is extracted from
%   W'CV   
%

% initialization
[nr,nc] = size(C);

% C * V = W * Bm
V = zeros(nc,m); % basis of Krylov subspace(right search space)
W = zeros(nr,m); % basis of C * V(left search space)
Vr = V; % residual of columns of V as approximated eigenvectors of C' * C
Bm = zeros(m,m);

% deflation if needed
if nargin == 14
	v = v - V_C * (V_C' * v);
	r = r - V_C * (V_C' * r);
end
V(:,1) = v; % v should be normalized
Vr(:,1) = r;
W(:,1) = C * V(:,1);
Bm(1,1) = sqrt(W(:,1)' * W(:,1));
W(:,1) = W(:,1) / Bm(1,1);
mvs = mvs + 1;

% Loop for Arnoldi iteration and Gram-Schmidt orthogalization
for i = 2:m-1                
    % Apply preconditioner if given 
    if ~isempty(L)
        if isnumeric(L) 
            r = L \ r;
            r = L' \ r;    
        else 
            r = feval(L,r); 
        end
        precs = precs + 1;
    end
    % orthogonalize r with V(:,1:i-1) 
    r = r - V(:,1:i-1) * (V(:,1:i-1)' * r);

    % reorthogonalization if m > 6 
    if( m > 6)  
         r = r - V(:,1:i-1) * (V(:,1:i-1)' * r); 
    end
    
    % deflation if needed
    if nargin == 14
        r = r - V_C * (V_C' * r);
    end
    
    % break if converged
    tmp = sqrt(r' * r);
    if tmp == 0 
        m = i;
        break; 
    end
    
    % normalize and save new basis vector 
    V(:,i) = r / tmp;
    W(:,i) = C * V(:,i);
    r = C' * W(:,i) - lambda * V(:,i);
    Vr(:, i) = r;  
    
    % number of matrix-vector multiplications
    mvs = mvs + 2;
    
    % orthogonalize W(:,i)
    Bm(1:i-1,i) = W(:,1:i-1)' * W(:,i);
    W(:,i) = W(:,i) - W(:,1:i-1) * Bm(1:i-1,i);
    % reorthogonalization if m > 6 
    if m > 6  
        W(:,i) = W(:,i) - W(:,1:i-1) * (W(:,1:i-1)' * W(:,i));
    end
    Bm(i,i) = sqrt(W(:,i)' * W(:,i));
    W(:,i) = W(:,i) / Bm(i,i);    
end 

% add the diff vector to the basis
diffNorm = sqrt(diff' * diff);
tmp = V(:,1:m-1)' * diff;
diff = diff - V(:,1:m-1) * tmp;
rDiff = rDiff - Vr(:,1:m-1) * tmp;

% reorthogonalization if m > 6
if m > 6
    tmp = V(:,1:m-1)' * diff;
    diff = diff - V(:,1:m-1) * tmp;
    rDiff = rDiff - Vr(:,1:m-1) * tmp;
end

% deflation if needed
if nargin == 14
	diff = diff - V_C *  (V_C' * diff);
	rDiff = rDiff - V_C * (V_C' * rDiff);        
end

tmp = sqrt(diff' * diff);

% check and add diff only if it's significant
if tmp <= 1e-8 * diffNorm
    % Apply preconditioner if given  
    if ~isempty(L)
        if isnumeric(L) 
            r = L \ r;
            r = L' \ r;    
        else 
            r = feval(L,r); 
        end
    end
    % orthogonalize r with V(:,1:m-1) 
    r = r - V(:,1:m-1) * (V(:,1:m-1)' * r);

    % reorthogonalization if m > 6 
    if( m > 6)  
         r = r - V(:,1:m-1) * (V(:,1:m-1)' * r); 
    end
    
    % deflation if needed
    if nargin == 14
        r = r - V_C * (V_C' * r);
    end
    
    % break if converged
    tmp = sqrt(r' * r);
    if tmp == 0 
        m = m - 1;
    else
        % normalize and save new basis vector 
        V(:,m) = r / tmp;
        W(:,m) = C * V(:,m);
        r = C' * W(:,m) - lambda * V(:,m);    
        Vr(:, m) = r;

        % orthogonalize W(:,i)
        Bm(1:m-1,m) = W(:,1:m-1)' * W(:,m);
        W(:,m) = W(:,m) - W(:,1:m-1) * Bm(1:m-1,i);
        % reorthogonalization if m > 6 
        if m > 6  
            W(:,m) = W(:,m) - W(:,1:m-1) * (W(:,1:m-1)' * W(:,m));
        end
        Bm(m,m) = sqrt(W(:,m)' * W(:,m));
        W(:,m) = W(:,i) / Bm(i,m);            
    end 
    mvs = mvs + 2;
elseif tmp <= 1e-2 * diffNorm
    % recompute residual of diff i.e. C' * C * diff - sigma ^ 2 * diff if necessary
    V(:,m) = diff / tmp;
    diff = V(:,m); 
    W(:,m) = C * V(:,m);
    rDiff = C' * W(:,m) - lambda * diff;     
    Vr(:,m) = rDiff;        
    mvs = mvs + 2;
    
%     orthogonalize W(:,m)
    Bm(1:m-1,m) = W(:,1:m-1)' * W(:,m);
    W(:,m) = W(:,m) - W(:,1:m-1) * Bm(1:m-1,m);
        
    % reorthogonalization if m > 6 
    if m > 6 
        W(:,m) = W(:,m) - W(:,1:m-1) * (W(:,1:m-1)' * W(:,m));
    end
    Bm(m,m) = sqrt(W(:,m)'*W(:,m));    
    W(:,m) = W(:,m) / Bm(m,m);    
else     
    V(:,m) = diff / tmp; 
    Vr(:,m) = rDiff / tmp;    
    W(:,m) = C * V(:,m);
    mvs = mvs + 1;
    
    % orthogonalize W(:,m)
    Bm(1:m-1,m) = W(:,1:m-1)' * W(:,m);
    W(:,m) = W(:,m) - W(:,1:m-1) * Bm(1:m-1,m);
        
    % reorthogonalization if m > 6 
    if m > 6 
        W(:,m) = W(:,m) - W(:,1:m-1) * (W(:,1:m-1)' * W(:,m));
    end
    Bm(m,m) = sqrt(W(:,m)'*W(:,m));    
    W(:,m) = W(:,m) / Bm(m,m);
end

% compute singular triplets of projected matrix
[U_B,S_B,V_B] = svd(Bm(1:m,1:m)); % S_D are with decreasing order
S_B = diag(S_B);
if FindMax == 0
    svidx = m;
else
    svidx = 1;
end
sigma = S_B(svidx); % approximate singular value
v = V(:,1:m) * V_B(:, svidx); % approximate right singular vector
u = W(:,1:m) * U_B(:,svidx); %

% get the residual of v as eigenvector of C' * C
r = Vr(:,1:m) * V_B(:,svidx);
tmp = sigma * sigma;
shift = tmp - lambda;
r = r - shift * v;

res = sqrt(r' * r); % residual of (sigma, v) as eigenpair of C' * C
if tripletres
   res = res / sigma; % residual of (sigma,u,v) as singular triplet
end

% get corresponding eigenvalue of C' * C
lambda = tmp;

% update new diff and related vectors for the next iteration   
V_B(1,svidx) = V_B(1,svidx) - 1;
diff = V(:,1:m) * V_B(1:m,svidx);
rDiff = Vr(:,1:m) * V_B(1:m,svidx); 
rDiff = rDiff - shift * diff;

 % recompute r if necessary 
if  m > 15        
    r = C' * (C * v);
    lambda = (v' * r) / (v' * v); 
    r = r - lambda * v;
    mvs = mvs + 2;
end

% use a new diff if converged. 
if res < tol
    diff = V(:,1:m) * V_B(:, svidx - 1 + 2 * FindMax);
end
 
function [U_C, S_C, V_C,residuals, mvs,precs] = ...
    ifreesvd(C, FindMax, m,shift,rifthresh,zrifthresh,rifnnz, ...
    tol, itermax, k, V,  outputYes,usePrecon, tripletres)
%
%  The main function that carries out the (outer) iteration. 
%    
%  To be consistent with the notation in the paper, here we use C to denote
%  the target matrix and A = C' * C.
%

[nr,nc] = size(C); % Get size of C

% Compute preconditioner of svdifp
if isscalar(usePrecon) && usePrecon == 1 && ~isempty(shift) && FindMax == 0
    if shift == 0 && rifthresh == 0
        if outputYes ~= 0
            fprintf('===================================================================\n');
            fprintf('Computing a precondioner using qr, if it takes too long, try using RIF by setting a nonzero opt.rifthresh\n');
            fprintf('====================================================================\n');
        end
        R = qr(C,0);
        for i = 1:nc
            if(R(i,i) == 0)
               R(i,i) = 1e-8; 
            end
        end
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
            L = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
        else
            fprintf('RIF.mexa32 or RIF.mexa64 not found; Matlab implementation RIFLDL is used instead\n');
            L = RIFLDL(C,shift,rifthresh,zrifthresh, rifnnz);
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

if startL ~= 0
    nnzL = nnz(L);
else 
    nnzL = 0;
end

% initialization 
S_C = zeros(k,1); % converged k singular values of C
V_C = zeros(nc,k); % converged k right singular vectors of C
U_C = zeros(nr,k); % converged k left singular vectors of C
residuals = zeros(k,1); % residuals of converged k singular triplets
mvs = 0; % number of matrix-vector multiplications
precs = 0; % number of actions of precondioners

conv = ones(itermax,1); % variables for adaptively choosing m
rate = ones(10,1);

initialConv = 0; % initialization of sign of lth singular triplet converges

diff = zeros(nc,1); % Initialization of diff vector       
rDiff = diff; % C' * C * diff - sigma ^ 2 * diff


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
    
    normv = sqrt(v' * v);
    if normv == 0
       error('IFREESVD: Invalid initial left singular vector'); 
    end
    
    v = v / normv; % normalize v
    u = C * v; % initial left singular vector
    lambda = u' * u; % rayleigh quotient of v, approximated eigenvalue of C' * C 
    sigma = sqrt(lambda);% approximate singular value of C, norm of u
    u = u / sigma; % normalize u
    r2 = C' * u;
    
    r2 = r2 - sigma * v; % r2 = C' * u - sigma * v;
    r = r2 * sigma; % residual of u as eigenvalue of C' * C    
    if tripletres
        res = sqrt(r' * r); % residual of u as eigenvalue of C' * C  
    else
        res = sqrt(r2' * r2); % residual of singular triplet 
    end
    
    mvs = mvs + 2;
        
    if l > 1
        diff = zeros(nc,1); % Initialization of diff vector       
        rDiff = diff; % C' * C * diff - sigma ^ 2 * diff
    end
    
    % parameters for adaptively choosing m
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
                    [sigma, lambda, v, u, r,  diff, rDiff, res, mvs,precs] = ...
              parnob(C, FindMax, [], mvs,precs, lambda, v, r,  diff, rDiff, projSize, tol,tripletres, V_C(:,1:l-1));
     
          else
              % with preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs,precs] = ...
              parnob(C, FindMax, L, mvs,precs, lambda, v, r,  diff, rDiff,  projSize, tol,tripletres, V_C(:,1:l-1));
                   
          end           
        else
          % no deflation if l = 1. 
          if ( startL == 0 )
              % without preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs,precs] = ...
              parnob(C, FindMax, [], mvs,precs, lambda, v, r,  diff, rDiff,  projSize, tol,tripletres);
          else
              % with preconditioning
                    [sigma, lambda, v, u, r,  diff, rDiff,  res, mvs,precs] = ...
              parnob(C, FindMax, L, mvs,precs, lambda, v, r,  diff, rDiff,  projSize, tol,tripletres);
                   
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

        % Check whether to compute preconditioner based on convergence behavior
        if isscalar(usePrecon) && usePrecon == 1 && (res <= tol * 10 || iter >= itermax / 5) && (startL == 0) && res > tol
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
                fprintf('RIF.mexa32 or RIF.mexa64 not found, Matlab implementation RIFLDL is used instead\n');
                L = RIFLDL(C,shift,rifthresh,zrifthresh, rifnnz);
            end
            if outputYes ~= 0
                fprintf('Done!\n');
                fprintf('====================================================================\n');
                fprintf('Start preconditioned SVDIFP\n');
            end
        end
        
        % check whether compute a new preconditioner based on convergence behavior
        if isscalar(usePrecon) && usePrecon == 1 && (((res >= sqrt(tol)) && iter >= itermax / 2) || ((res >= tol * 1e2) && iter >=itermax - 50))...
                && Lcomputed <= 1
            Lcomputed = 2;
            shift = sigma;
            if outputYes ~= 0
                fprintf('===================================================================\n');
                fprintf('ReComputing RIF of matrix A using shift %f, if it takes too long, try increasing opt.rifthresh\n',shift);
            end
            if RIFexist == 3
                L = RIF(C,shift,rifthresh,zrifthresh,rifnnz);
            else
                fprintf('RIF.mexa32 or RIF.mexa64 not found, Matlab implementation RIFLDL is used instead\n');
                L = RIFLDL(C,shift,rifthresh,zrifthresh, rifnnz);
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
    V_C(:,l) = v / norm(v);
    U_C(:,l) = u / norm(u);
    residuals(l) = res;
    
    % warn if not converged      
    if res >= tol && outputYes ~= 0
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

function L = RIFLDL(A,shift,rifthresh,zrifthresh, rifnnz)
%
% Robust Incomplete factorization of A'A
%

[~,n] = size(A); % get number of columns of A
nnzmax = min(rifnnz*n, n*(n+1)/2);
try
    L = spalloc(n,n,nnzmax); % preallocate space for L
catch err
    error('RIFLDL: Out of memory! Try to input a smaller opt.rifnnz!');
end
Z = speye(n,n);
threshold = rifthresh * sum(abs(A))';
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
    try
        L(j+1:n,j) = tmpcol;
    catch err
        error('Exceeded Preallocated memory! Try using a bigger nnzmax!');
    end
    tmpcoef = tmpcol / L(j,j);    
    % Update columns of Z   
    Z = Z(:,2:end) - Z(:,1) * tmpcoef';
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
