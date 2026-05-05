import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;

// 图片数据加载器：从目录批量读取图片并转换为 CNN 所需的灰度矩阵
// 支持两种目录结构：
//   平铺模式：images/0_rat.jpg, 1_ox.jpg ...  文件名首字符为类别编号
//   子目录模式：images/0/rat1.jpg, images/1/ox1.jpg ...  子目录名为类别编号（推荐）
// 核心原理：
//   灰度转换：gray = 0.299*R + 0.587*G + 0.114*B（ITU-R BT.601 加权公式，模拟人眼对绿色最敏感）
//   归一化：像素值除以 255，压缩到 [0,1]，防止数值过大影响梯度稳定性
//   迭代器模式：hasNext/next/reset 控制按需逐张读取，避免一次性加载所有图片占用大量内存
public class DataLoader {
    private File[]  imageFiles;  // 所有图片文件（跨子目录展平后的列表）
    private int[]   labels;      // 每张图片对应的类别编号，与 imageFiles 一一对应
    private int     currentIndex;
    private final boolean normalize;

    // 构造函数：自动检测目录结构，选择对应解析方式
    DataLoader(String directoryPath, boolean normalize) {
        File dir = new File(directoryPath);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IllegalArgumentException("路径不存在或不是文件夹: " + directoryPath);
        }
        this.normalize = normalize;
        this.currentIndex = 0;

        // 检查是否存在数字命名的子目录（子目录模式）
        File[] subDirs = dir.listFiles(f -> f.isDirectory() && f.getName().matches("\\d+"));
        if (subDirs != null && subDirs.length > 0) {
            loadFromSubDirectories(dir);  // 子目录模式：images/0/, images/1/, ...
        } else {
            loadFromFlatDirectory(dir);   // 平铺模式：images/0_rat.jpg, 1_ox.jpg, ...
        }
    }

    // 子目录模式：子目录名即为类别编号
    // images/0/ → label=0，images/1/ → label=1，以此类推
    private void loadFromSubDirectories(File dir) {
        File[] subDirs = dir.listFiles(File::isDirectory);
        Arrays.sort(subDirs); // 按目录名排序，保证每次顺序一致

        List<File>    fileList  = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        for (File subDir : subDirs) {
            int label;
            try {
                label = Integer.parseInt(subDir.getName()); // 目录名转类别编号
            } catch (NumberFormatException e) {
                continue; // 跳过非数字命名的目录
            }
            File[] imgs = subDir.listFiles((d, name) -> isImageFile(name));
            if (imgs == null) continue;
            Arrays.sort(imgs); // 同目录内按文件名排序
            for (File img : imgs) {
                fileList.add(img);
                labelList.add(label);
            }
        }

        imageFiles = fileList.toArray(new File[0]);
        labels = labelList.stream().mapToInt(Integer::intValue).toArray();
    }

    // 平铺模式：从文件名前缀解析类别编号，格式为 "类别编号_文件名.扩展名"
    // 例：0_rat.jpg → label=0，11_pig2.jpg → label=11
    private void loadFromFlatDirectory(File dir) {
        File[] imgs = dir.listFiles((d, name) -> isImageFile(name));
        if (imgs == null) imgs = new File[0];
        Arrays.sort(imgs);

        List<File>    fileList  = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        for (File img : imgs) {
            String name = img.getName();
            int underscoreIdx = name.indexOf('_');
            if (underscoreIdx < 0) {
                throw new IllegalArgumentException(
                    "平铺模式文件名必须以 '类别编号_' 开头，当前文件名不符合：" + name);
            }
            try {
                int label = Integer.parseInt(name.substring(0, underscoreIdx)) - 1; // 文件名从1开始，转为0索引
                fileList.add(img);
                labelList.add(label);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("无法从文件名解析类别编号：" + name);
            }
        }

        imageFiles = fileList.toArray(new File[0]);
        labels = labelList.stream().mapToInt(Integer::intValue).toArray();
    }

    private static boolean isImageFile(String name) {
        String lower = name.toLowerCase();
        return lower.endsWith(".jpg") || lower.endsWith(".jpeg")
                || lower.endsWith(".png") || lower.endsWith(".bmp");
    }

    public boolean hasNext()       { return currentIndex < imageFiles.length; }
    public int size()               { return imageFiles.length; }
    public void reset()             { currentIndex = 0; }
    public String currentFileName() { return currentIndex < imageFiles.length ? imageFiles[currentIndex].getName() : null; }

    // 返回当前图片的真实类别编号
    public int currentLabel() {
        if (currentIndex >= imageFiles.length) throw new IllegalStateException("已超出图片范围");
        return labels[currentIndex];
    }

    // 读取下一张图片，返回 double[1][height][width]（灰度单通道，第一维固定为1）
    public double[][][] next() throws IOException {
        if (!hasNext()) throw new IllegalStateException("没有更多图片了");
        double[][][] img = loadImage(imageFiles[currentIndex]);
        currentIndex++;
        return img;
    }

    // 将目录中所有图片依次送入 CNN 前向传播，返回所有图片的预测结果
    public double[][] feedAll(CNN cnn) throws IOException {
        reset();
        double[][] results = new double[imageFiles.length][];
        for (int i = 0; i < imageFiles.length; i++) {
            System.out.println("正在处理: " + imageFiles[i].getName());
            results[i] = cnn.forward(next());
        }
        return results;
    }

    // 读取单张图片并转为灰度矩阵
    // getRGB 返回 ARGB 整数，通过位运算提取 R/G/B 各8位
    private double[][][] loadImage(File file) throws IOException {
        BufferedImage img = ImageIO.read(file);
        if (img == null) throw new IOException("无法读取图片: " + file.getName());

        int h = img.getHeight();
        int w = img.getWidth();
        double[][] channel = new double[h][w];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int rgb = img.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8)  & 0xFF;
                int b =  rgb        & 0xFF;
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                channel[i][j] = normalize ? gray / 255.0 : gray;
            }
        }
        return new double[][][]{channel};
    }
}
