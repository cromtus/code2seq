package JavaExtractor.Common;

import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<Node> treeAsSequence;
    private final String name;

    private final String content;

    public MethodContent(ArrayList<Node> treeAsSequence, String name, String content) {
        this.treeAsSequence = treeAsSequence;
        this.name = name;
        this.content = content;
    }

    public ArrayList<Node> getTreeAsSequence() {
        return treeAsSequence;
    }

    public String getName() {
        return name;
    }

    public String getContent() {
        return content;
    }
}
