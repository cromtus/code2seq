package JavaExtractor.FeaturesEntities;

public class ProgramFeatures {
    private String name, serializedTree;

    public ProgramFeatures(String name, String serializedTree) {
        this.name = name;
        this.serializedTree = serializedTree;
    }

    @SuppressWarnings("StringBufferReplaceableByString")
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append(" ").append(serializedTree);
        return stringBuilder.toString();
    }
}
