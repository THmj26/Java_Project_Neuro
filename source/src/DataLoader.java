import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class DataLoader {
    private File[] imageFiles;
    private int currentIndex;
    private final boolean normalize;

    DataLoader(String directoryPath, boolean normalize) {
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

    public String currentFileName() {
        if (currentIndex < imageFiles.length)
            return imageFiles[currentIndex].getName();
        return null;
    }

    // 读取下一张图片，返回 double[1][height][width]（灰度，1通道）
    public double[][][] next() throws IOException {
        if (!hasNext()) throw new IllegalStateException("没有更多图片了");
        double[][][] img = loadImage(imageFiles[currentIndex]);
        currentIndex++;
        return img;
    }

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

        double[][] channel = new double[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int rgb = img.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                channel[i][j] = normalize ? gray / 255.0 : gray;
            }
        }
        return new double[][][]{channel};
    }
}
