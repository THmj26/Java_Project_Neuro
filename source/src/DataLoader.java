import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class DataLoader {
    private File[] imageFiles;
    private int currentIndex;
    private final boolean grayscale;
    private final boolean normalize;

    DataLoader(String directoryPath, boolean grayscale, boolean normalize) {
        File dir = new File(directoryPath);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IllegalArgumentException("路径不存在或不是文件夹: " + directoryPath);
        }
        imageFiles = dir.listFiles((d, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".jpg") || lower.endsWith(".jpeg")
                    || lower.endsWith(".png") || lower.endsWith(".bmp");
        });
        if (imageFiles == null) imageFiles = new File[0];
        Arrays.sort(imageFiles);
        this.grayscale = grayscale;
        this.normalize = normalize;
        this.currentIndex = 0;
    }

    public boolean hasNext() {
        return currentIndex < imageFiles.length;
    }

    public int size() {
        return imageFiles.length;
    }

    public void reset() {
        currentIndex = 0;
    }

    // 返回当前待读取图片的文件名（未读取时可用于日志）
    public String currentFileName() {
        if (currentIndex < imageFiles.length)
            return imageFiles[currentIndex].getName();
        return null;
    }

    // 读取下一张图片，返回 double[channels][height][width]
    public double[][][] next() throws IOException {
        if (!hasNext()) throw new IllegalStateException("没有更多图片了");
        double[][][] img = loadImage(imageFiles[currentIndex]);
        currentIndex++;
        return img;
    }

    // 将文件夹中所有图片逐一喂给 CNN，返回每张图片对应的输出
    public double[][] feedAll(CNN cnn) throws IOException {
        reset();
        double[][] results = new double[imageFiles.length][];
        for (int i = 0; i < imageFiles.length; i++) {
            System.out.println("正在处理: " + imageFiles[i].getName());
            results[i] = cnn.forward(next());
        }
        return results;
    }

    private double[][][] loadImage(File file) throws IOException {
        BufferedImage img = ImageIO.read(file);
        if (img == null) throw new IOException("无法读取图片: " + file.getName());

        int h = img.getHeight();
        int w = img.getWidth();

        if (grayscale) {
            double[][] channel = new double[h][w];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int rgb = img.getRGB(j, i);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    // 标准灰度公式
                    double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    channel[i][j] = normalize ? gray / 255.0 : gray;
                }
            }
            return new double[][][]{channel};
        } else {
            double[][] red   = new double[h][w];
            double[][] green = new double[h][w];
            double[][] blue  = new double[h][w];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int rgb = img.getRGB(j, i);
                    double r  = (rgb >> 16) & 0xFF;
                    double gv = (rgb >> 8)  & 0xFF;
                    double bv =  rgb        & 0xFF;
                    red[i][j]   = normalize ? r  / 255.0 : r;
                    green[i][j] = normalize ? gv / 255.0 : gv;
                    blue[i][j]  = normalize ? bv / 255.0 : bv;
                }
            }
            // 顺序: R、G、B 三通道，与 convNeuron 的 kernel 通道对应
            return new double[][][]{red, green, blue};
        }
    }
}
