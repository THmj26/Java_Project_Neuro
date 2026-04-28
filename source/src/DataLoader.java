import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

// 文件：DataLoader.java
// 功能：从指定目录批量读取图片，转换为灰度矩阵，供CNN前向传播使用
public class DataLoader {
    private File[] imageFiles;  // 目录下所有图片文件
    private int currentIndex;   // 当前读取位置（迭代器模式）
    private final boolean normalize; // 是否将像素值归一化到 [0,1]

    // 构造函数：扫描指定目录下的所有图片文件，按文件名排序
    DataLoader(String directoryPath, boolean normalize) {
        File dir = new File(directoryPath);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IllegalArgumentException("路径不存在或不是文件夹: " + directoryPath);
        }
        // 只读取 jpg/jpeg/png/bmp 格式
        imageFiles = dir.listFiles((d, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".jpg") || lower.endsWith(".jpeg")
                    || lower.endsWith(".png") || lower.endsWith(".bmp");
        });
        if (imageFiles == null) imageFiles = new File[0];
        // 按文件名字母顺序排序，保证每次读取顺序一致
        Arrays.sort(imageFiles);
        this.normalize = normalize;
        this.currentIndex = 0;
    }

    // 是否还有未读取的图片
    public boolean hasNext() {
        return currentIndex < imageFiles.length;
    }

    // 返回目录下图片总数
    public int size() {
        return imageFiles.length;
    }

    // 重置读取位置到第一张图片
    public void reset() {
        currentIndex = 0;
    }

    // 返回当前待读取图片的文件名（用于日志输出）
    public String currentFileName() {
        if (currentIndex < imageFiles.length)
            return imageFiles[currentIndex].getName();
        return null;
    }

    // 读取下一张图片，返回 double[1][height][width]（灰度单通道）
    // 每次调用后自动将指针移到下一张
    public double[][][] next() throws IOException {
        if (!hasNext()) throw new IllegalStateException("没有更多图片了");
        double[][][] img = loadImage(imageFiles[currentIndex]);
        currentIndex++;
        return img;
    }

    // 将目录中所有图片依次喂给CNN，返回每张图片对应的前向传播输出
    public double[][] feedAll(CNN cnn) throws IOException {
        reset();
        double[][] results = new double[imageFiles.length][];
        for (int i = 0; i < imageFiles.length; i++) {
            System.out.println("正在处理: " + imageFiles[i].getName());
            results[i] = cnn.forward(next());
        }
        return results;
    }

    // 将图片文件读取并转换为灰度矩阵
    // 灰度公式：gray = 0.299*R + 0.587*G + 0.114*B（ITU-R BT.601标准）
    // 返回 double[1][height][width]，通道数固定为1
    private double[][][] loadImage(File file) throws IOException {
        BufferedImage img = ImageIO.read(file);
        if (img == null) throw new IOException("无法读取图片: " + file.getName());

        int h = img.getHeight();
        int w = img.getWidth();

        double[][] channel = new double[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                // getRGB 返回 ARGB 格式的整数，通过位运算提取各通道
                int rgb = img.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8)  & 0xFF;
                int b =  rgb        & 0xFF;
                // 加权平均转灰度
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                // normalize=true 时缩放到 [0,1]，否则保留原始 [0,255]
                channel[i][j] = normalize ? gray / 255.0 : gray;
            }
        }
        // 包装成三维数组返回，第一维是通道数（灰度固定为1）
        return new double[][][]{channel};
    }
}
