package JavaExtractor.Common;

import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<Node> treeSequence;
    private final ArrayList<Integer> parentIndices;
    private final String name;

    private final String content;

    public MethodContent(
            ArrayList<Node> treeSequence,
            ArrayList<Integer> parentIndices,
            String name,
            String content
    ) {
        this.treeSequence = treeSequence;
        this.parentIndices = parentIndices;
        this.name = name;
        this.content = content;
    }

    public ArrayList<Node> getTreeSequence() {
        return treeSequence;
    }

    public ArrayList<Integer> getParentIndices() {
        return parentIndices;
    }

    public String getName() {
        return name;
    }

    public String getContent() {
        return content;
    }
}
