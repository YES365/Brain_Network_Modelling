function plot_volume_mapping(volumeFile,surfaceFile,outputFile,optionFile)

% Matlab script for BrainNetViewer Volume Mapping

% 读取volume数据以获取数据范围
volume_data = niftiread(volumeFile);
max_value = max(volume_data(:));
min_value = min(volume_data(:));

% % 启动BrainNet Viewer
% BrainNet;
% 
% % 等待BrainNet Viewer完全加载
% pause(2);
% 
% % 加载Surface文件和Volume文件
% BrainNet_MapCfg('SurfaceFile', surfaceFile, ...
%                       'VolumeFile', volumeFile);
% 
% % 设置视图为Lateral, medial and dorsal view
% view_option = 3;  % 对应Lateral, medial and dorsal view
% BrainNet_Option('Layout', 'Custom');
% BrainNet_Option('CustomView', view_option);
% 
% % 设置Volume mapping显示选项
% BrainNet_Option('VolumeMappingDisplay', 'Positive & Negative');
% 
% % 设置Positive和Negative范围
% BrainNet_Option('VolumeMappingPositiveRange', [0 max_value]);
% BrainNet_Option('VolumeMappingNegativeRange', [min_value 0]);
% 
% % 设置Map algorithm为Most Neighbor Voxel
% BrainNet_Option('VolumeMappingAlgorithm', 'Most Neighbor Voxel');
% 
% % 设置Colormap为Jet
% BrainNet_Option('ColorMap', 'jet');
% 
% % 更新显示
% BrainNet_Option('Update');

% % 设置BrainNet Viewer的参数
% options = struct();
% options.layout = 'LMR'; % Lateral, medial and dorsal view
% options.display = 'Both'; % Positive & Negative
% options.positive_range = [0, max_value ];
% options.negative_range = [min_value, 0];
% options.mapalgorithm = 'MN'; % Most Neighbor Voxel
% options.colormap = 'jet';
% BrainNet_Option(options)

% 调用BrainNet Viewer绘制图像
load(optionFile)
EC.vol.px=max_value;
EC.vol.nx=min_value;
save(optionFile,"EC")

BrainNet_MapCfg(surfaceFile, volumeFile,optionFile);

% 保存图像为png格式
print('-dpng', outputFile);

end