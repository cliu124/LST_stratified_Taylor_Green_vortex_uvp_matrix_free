clear;
close all;
clc;

solve_eig=1;%if solve_eig=1, it will solve eigenvalue problem. Otherwise, it will jump to post-processing
save_results=1;%whether we want to save results. Note this may overwrite existing data
post_growth_rate=1;%post-processing of growth rate
post_eigenvector=1;%post-processing of eigenvectors

params.Re=1600;%Reynolds number
params.Pr=0.7;%Prandtl number

Fr_list=[1,0.5,0.25,0.125];
kz_list=1:1:19;

%resolution in x and y
params.Nx=64;
params.Ny=64;
params.N=params.Nx*params.Ny;

%domain size. In default, they are both 2pi
params.Lx=2*pi;
params.Ly=2*pi;

% One-dimensional Fourier differentiation matrices.
[x,D1x]=fourdif(params.Nx,1);
[~,D2x]=fourdif(params.Nx,2);
params.Dx=D1x*(2*pi/params.Lx);
params.Dxx=D2x*(2*pi/params.Lx)^2;
params.x=x/(2*pi)*params.Lx;

[y,D1y]=fourdif(params.Ny,1);
[~,D2y]=fourdif(params.Ny,2);
params.Dy=D1y*(2*pi/params.Ly);
params.Dyy=D2y*(2*pi/params.Ly)^2;
params.y=y/(2*pi)*params.Ly;

% Taylor-Green base flow and its gradients.
[X,Y]=meshgrid(params.x,params.y);
params.U=sin(X).*cos(Y);
params.V=-cos(X).*sin(Y);
params.dUdx=cos(X).*cos(Y);
params.dUdy=-sin(X).*sin(Y);
params.dVdx=sin(X).*sin(Y);
params.dVdy=-cos(X).*cos(Y);

stationary_tolerance=1e-6;
num_eigenvalues=6;

outer_options.tol=1e-6;
outer_options.maxit=600;
outer_options.p=50;
outer_options.disp=0; %whether you want to display all iterations
outer_options.isreal=false;
outer_options.issym=false;

if solve_eig
    for kz_ind=1:length(kz_list)
        params.kz=kz_list(kz_ind);
        
        % Delta=dxx+dyy-kz^2 changes with kz, so its Fourier-mode LU factors
        % must be rebuilt whenever kz changes.  The factors do not depend on Fr
        % and are reused for every Froude number at this kz.
        params=add_laplacian_inverse(params);
        
        for Fr_ind=1:length(Fr_list)
            params.Fr=Fr_list(Fr_ind);
            solve_options=outer_options;
            
            % Seed eigs with the leading retained Ritz vector from the previous
            % kz.
            if kz_ind>1 && ~isempty(eigvec{Fr_ind,kz_ind-1})
                solve_options.v0=eigvec{Fr_ind,kz_ind-1}(:,1);
            else
                solve_options.v0=randn(3*params.N,1)+1i*randn(3*params.N,1);
            end
            solve_options.v0=solve_options.v0/norm(solve_options.v0);
            
            %the core eigs solver.
            [eigvec{Fr_ind,kz_ind},eigval{Fr_ind,kz_ind},eig_flag]=eigs( ...
                @(q) apply_A(q,params),3*params.N,num_eigenvalues, ...
                'largestreal',solve_options);
            
            fprintf('Completed Fr=%g, kz=%g, eigs flag=%d.\n', ...
                params.Fr,params.kz,eig_flag);
        end
    end
    
    if save_results
        save(['results_Re=',num2str(params.Re),'_Pr=',num2str(params.Pr),'.mat'],'eigval','eigvec','flag');
    end
end

if post_growth_rate
    load(['results_Re=',num2str(params.Re),'_Pr=',num2str(params.Pr),'.mat'],'eigval','eigvec','flag');
    
    % Post-process all retained modes. Figure 2(a) follows the stationary
    % branch, for which imag(sigma)=0. Nothing returned by eigs is discarded.
    for Fr_ind=1:length(Fr_list)
        for kz_ind=1:length(kz_list)
            params.Fr=Fr_list(Fr_ind);
            params.kz=kz_list(kz_ind);
            lambda_all=diag(eigval{Fr_ind,kz_ind});
            
            %identify the stationary modes.
            stationary_indices=find( ...
                abs(imag(lambda_all))<stationary_tolerance);
            
            if isempty(stationary_indices)
                warning(['No stationary eigenvalue found at Fr=%g, kz=%g. ' ...
                    'Using the retained mode with largest real part for plots.'], ...
                    params.Fr,params.kz);
                [~,selected_index]=max(real(lambda_all));
            else
                %select the mode with the largest growth rate among
                %stationary mode.
                [~,index_within_stationary]=max( ...
                    real(lambda_all(stationary_indices)));
                selected_index=stationary_indices(index_within_stationary);
            end
            
            stationary_mode_index{Fr_ind,kz_ind}=selected_index;
            stationary_eigval{Fr_ind,kz_ind}=lambda_all(selected_index);
            stationary_eigvec{Fr_ind,kz_ind}= ...
                eigvec{Fr_ind,kz_ind}(:,selected_index);
            
            q=stationary_eigvec{Fr_ind,kz_ind};
            lambda=stationary_eigval{Fr_ind,kz_ind};
        end
    end
    
    %plot the growth rate
    clear data plot_config
    for Fr_ind=1:length(Fr_list)
        growth_rate=real(cell2mat(stationary_eigval));
        data{Fr_ind}.x=kz_list;
        data{Fr_ind}.y=growth_rate(Fr_ind,:);
    end
    data=GTZ2024(data); %Add the data from Guo Taylor, Zhou (2024) JFM paper
    
    if length(Fr_list)>1
        plot_config.legend_list={1};
        for Fr_ind=1:length(Fr_list)
            Fr=Fr_list(Fr_ind);
            plot_config.legend_list{Fr_ind+1}=['Fr=',num2str(Fr)];
        end
    end
    plot_config.label_list={1,'$k_z$','$\sigma_r$'};
    plot_config.Markerindex=3;
    plot_config.fontsize_legend=34;
    plot_config.user_color_style_marker_list=...
        {'-b','--r','-.k',':m','bdiamond','r^','*k','msquare'};
    plot_config.name='growth_rate_validation_GTZ2024.png';
    plot_line(data,plot_config);
end

if post_eigenvector
    clear data plot_config;
    data{1}.x=params.x;
    data{1}.y=params.y;
    plot_config.label_list={1,'$x$','$y$'};
    variable_list={'u','v','rho','w','omega_z'};
    for Fr_ind=1:length(Fr_list)
        for kz_ind=1:length(kz_list)
            Fr=Fr_list(Fr_ind);
            kz=kz_list(kz_ind);
            selected_vector=eigvec{Fr_ind,kz_ind};
            eigvec_mat{Fr_ind,kz_ind}.u=reshape(selected_vector(1:params.N),[params.Ny,params.Nx]);
            eigvec_mat{Fr_ind,kz_ind}.v=reshape(selected_vector(params.N+1:2*params.N),[params.Ny,params.Nx]);
            eigvec_mat{Fr_ind,kz_ind}.rho=reshape(selected_vector(2*params.N+1:3*params.N),[params.Ny,params.Nx]);
            eigvec_mat{Fr_ind,kz_ind}.w=-1/(1i*params.kz)*(dx(eigvec_mat{Fr_ind,kz_ind}.u,params)+dy(eigvec_mat{Fr_ind,kz_ind}.v,params));
            eigvec_mat{Fr_ind,kz_ind}.omega_z=dx(eigvec_mat{Fr_ind,kz_ind}.v,params)-dy(eigvec_mat{Fr_ind,kz_ind}.u,params);
            
            for variable_ind=1:length(variable_list)
                variable=variable_list{variable_ind};
                data{1}.z=real(eigvec_mat{Fr_ind,kz_ind}.(variable));
                plot_config.name=['eigenvector_Fr=',num2str(Fr),'_kz=',num2str(kz),'_Re=',num2str(params.Re),'_Pr=',num2str(params.Pr),'_',variable,'.png'];
                plot_config.print_size=[1,900,800];
                if strcmp(variable,'rho')
                    plot_config.title_list={1,'$\rho$'};
                elseif strcmp(variable,'omega_z')
                    plot_config.title_list={1,'$\omega_z$'};
                else
                    plot_config.title_list={1,['$',variable,'$']};
                end
                plot_config.fontsize=28;
                plot_contour(data,plot_config);
            end
        end
    end
    
end

function y=apply_A(q,params)
% Apply B^(-1)*A to q. B is fixed and nonsingular for kz ~= 0.
% B matrix is blkdiag(laplacian, laplacian, I)
%Here B is the LHS of the (2.10) of Guo J, Taylor JR, Zhou Q. Zigzag instability of columnar Taylor–Green vortices in a strongly stratified fluid. Journal of Fluid Mechanics. 2024 Oct;997:A34.

rhs=apply_raw_A(q,params);
N=params.N;

u=laplacian_1d_fft_solve(rhs(1:N),params);
v=laplacian_1d_fft_solve(rhs(N+1:2*N),params);
% The third diagonal block of B is the identity.
rho=rhs(2*N+1:3*N);
y=[u;v;rho];
end

function params=add_laplacian_inverse(params)
% Factor Dyy-(kx^2+kz^2)*I for every discrete Fourier mode in x.
% This will be much faster and the just perform inverse Laplacian in each
% Fourier mode.
Nx=params.Nx;
Ny=params.Ny;
wave=fourier_wavenumbers(Nx);
params.laplacian_1d_factors=cell(Nx,1);

for kx_ind=1:Nx
    kx=(2*pi/params.Lx)*wave(kx_ind);
    laplacian_k=params.Dyy-(kx^2+params.kz^2)*eye(Ny);
    params.laplacian_1d_factors{kx_ind}= ...
        decomposition(laplacian_k,'lu');
end

end


function z=laplacian_1d_fft_solve(rhs,params)
% Apply Delta^(-1): FFT in x, solve in y mode by mode, then inverse FFT.

Ny=params.Ny;
Nx=params.Nx;
rhs_hat=fft(reshape(rhs,Ny,Nx),[],2);
z_hat=zeros(Ny,Nx,'like',rhs_hat);

for kx_ind=1:Nx
    z_hat(:,kx_ind)= ...
        params.laplacian_1d_factors{kx_ind}\rhs_hat(:,kx_ind);
end

z=reshape(ifft(z_hat,[],2),Ny*Nx,1);

end

function wave=fourier_wavenumbers(Nx)
% Wavenumber ordering used by MATLAB's FFT and fourdif.

N1=floor((Nx-1)/2);
N2=(-Nx/2)*ones(rem(Nx+1,2));
wave=[0:N1 N2 -N1:-1].';
end

function Aq=apply_raw_A(q,params)
% Matrix-free action of the original generalized-eigenproblem matrix A.
%This is applying the L matrix based on (2.11) of Guo J, Taylor JR, Zhou Q. Zigzag instability of columnar Taylor–Green vortices in a strongly stratified fluid. Journal of Fluid Mechanics. 2024 Oct;997:A34.

N=params.N;
Ny=params.Ny;
Nx=params.Nx;

u=reshape(q(1:N),Ny,Nx);
v=reshape(q(N+1:2*N),Ny,Nx);
rho=reshape(q(2*N+1:3*N),Ny,Nx);

Adv_u=advection(u,params);
Adv_v=advection(v,params);
Adv_rho=advection(rho,params);

% L_uu*u + L_uv*v + L_u_rho*rho, equations (2.11b)-(2.11d).
f_uu=Adv_u+params.dUdx.*u;
f_uv=params.dUdy.*v;
A_u=-laplacian(f_uu,params)+dxx(f_uu,params) ...
    +dxy(params.dVdx.*u,params) ...
    -dx(advection(dx(u,params),params),params) ...
    +laplacian(laplacian(u,params),params)/params.Re ...
    -laplacian(f_uv,params)+dxx(f_uv,params) ...
    +dxy(params.dVdy.*v+Adv_v,params) ...
    -dx(advection(dy(v,params),params),params) ...
    +(1i*params.kz/params.Fr^2)*dx(rho,params);

% L_vu*u + L_vv*v + L_v_rho*rho, equations (2.11e)-(2.11g).
f_vu=params.dVdx.*u;
f_vv=params.dVdy.*v+Adv_v;
A_v=-laplacian(f_vu,params)+dyy(f_vu,params) ...
    +dyx(params.dUdx.*u+Adv_u,params) ...
    -dy(advection(dx(u,params),params),params) ...
    -laplacian(f_vv,params)+dyy(f_vv,params) ...
    +dxy(params.dUdy.*v,params) ...
    -dy(advection(dy(v,params),params),params) ...
    +laplacian(laplacian(v,params),params)/params.Re ...
    +(1i*params.kz/params.Fr^2)*dy(rho,params);

% Density row, equations (2.11h)-(2.11j).
A_rho=-(1/(1i*params.kz))*dx(u,params) ...
    -(1/(1i*params.kz))*dy(v,params) ...
    -Adv_rho ...
    +laplacian(rho,params)/(params.Re*params.Pr);

Aq=[A_u(:);A_v(:);A_rho(:)];
end

function value=advection(f,params)
% U*df/dx+V*df/dy.
value=params.U.*dx(f,params)+params.V.*dy(f,params);
end

function value=laplacian(f,params)
% Delta*f=(dxx+dyy-kz^2)*f.
value=dxx(f,params)+dyy(f,params)-params.kz^2*f;
end

function value=dx(f,params)
value=f*params.Dx.';
end

function value=dy(f,params)
value=params.Dy*f;
end

function value=dxx(f,params)
value=f*params.Dxx.';
end

function value=dyy(f,params)
value=params.Dyy*f;
end

function value=dxy(f,params)
value=dx(dy(f,params),params);
end

function value=dyx(f,params)
value=dy(dx(f,params),params);
end

function data=GTZ2024(data)
data_len=length(data);

%digitized data from figure 2(a) of Guo J, Taylor JR, Zhou Q. Zigzag instability of columnar Taylor–Green vortices in a strongly stratified fluid. Journal of Fluid Mechanics. 2024 Oct;997:A34.

%Fr=1;
growth_rate_Fr1=[1	0.1809
    2	0.2108
    3	0.2105
    4	0.2032
    5	0.1931
    6	0.1818
    7	0.1671
    8	0.1516
    9	0.1343
    10	0.1153
    11	0.0944
    12	0.0713
    13	0.0462
    14	0.0186
    ];
data{data_len+1}.x=growth_rate_Fr1(:,1);
data{data_len+1}.y=growth_rate_Fr1(:,2);

%Fr=0.5
growth_rate_Fr0p5=[1	0.1429
    2	0.2064
    3	0.2251
    4	0.2202
    5	0.2122
    6	0.1993
    7	0.1859
    8	0.1710
    9	0.1559
    10	0.1401
    11	0.1243
    12	0.1096
    13	0.0939
    14	0.0763
    15	0.0567
    ];
data{data_len+2}.x=growth_rate_Fr0p5(:,1);
data{data_len+2}.y=growth_rate_Fr0p5(:,2);

%Fr=0.25
growth_rate_Fr0p25=[1	0.0775
    2	0.1383
    3	0.1902
    4	0.2269
    5	0.2422
    6	0.2333
    7	0.2120
    8	0.2026
    9	0.1871
    10	0.1680
    11	0.1508
    12	0.1325
    13	0.1123
    14	0.0916
    15	0.0693
    16	0.0459
    17	0.0209
    ];
data{data_len+3}.x=growth_rate_Fr0p25(:,1);
data{data_len+3}.y=growth_rate_Fr0p25(:,2);

%Fr=0.125
growth_rate_Fr0p125=[1	0.0375
    2	0.0699
    3	0.1006
    4	0.1277
    5	0.1508
    6	0.1700
    7	0.1852
    8	0.1953
    9	0.1998
    10	0.1981
    11	0.1902
    12	0.1767
    13	0.1559
    14	0.1267
    15	0.0930
    16	0.0626
    17	0.0426
    18	0.0256
    19	0.0045
    ];

data{data_len+4}.x=growth_rate_Fr0p125(:,1);
data{data_len+4}.y=growth_rate_Fr0p125(:,2);


end