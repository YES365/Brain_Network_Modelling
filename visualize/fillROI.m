function fillROI(mask,region,value,output)
%% Fill Values into ROIs Based on A Given ROI Mask
% 
%  Version 1.0.0 - Z.K.X. 2018/10/12
%
%---------------------------------------------------------------------------------------------%
%% Input
%  mask: mask file with labeled ROI serial numbers
%  region: ROI labels you want to fill
%  value: values you want to fill into ROIs
%  output: output path and file name
%---------------------------------------------------------------------------------------------%
%% Dependency
%  Tools for NIfTI and ANALYZE image
%  Load, save, make, reslice, view (and edit) both NIfTI and ANALYZE data on any platform
%  Download Link:
%  http://cn.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% ---------------------------------------------------------------------------------------------%

if (nargin < 3) 
	output = 'filled_ROI.nii';
end

% b = load_nii('C:\Users\77293\OneDrive\My_Files\My_Study\My_Research\Postgraduate_Projects\code_base\atlas\my_atlas\desikanKilliany_MNI_2mm_89.nii');

T = niftiread(mask);
T_info = niftiinfo(mask);
T_info.Datatype = 'double';
C = double(T);
% C.hdr = b.hdr;

C(~ismember(C,region)) = 0;
for i = 1:length(value)
    C(C==region(i)) = value(i);
end

niftiwrite(C,output,T_info)

% GPT的答案
% 您可以尝试在保存nifti文件之前更改hdr信息中的数据类型。下面是一个示例，它将hdr信息中的数据类型从’uint8’更改为’double’，然后保存nifti文件：
% 
% % 加载nifti文件
% nii = load_nii('your_file.nii');
% img = nii.img;
% 
% % 将图像数据从'uint8'转换为'double'
% img = double(img);
% 
% % 更新nifti结构中的图像数据
% nii.img = img;
% 
% % 更改hdr信息中的数据类型
% nii.hdr.dime.datatype = 64; % 64代表双精度浮点数（double）
% nii.hdr.dime.bitpix = 64; % 每个像素用64位存储
% 
% % 保存更改后的nifti文件
% save_nii(nii, 'new_file.nii');
% Copy
% 希望这个示例能够解决您的问题。