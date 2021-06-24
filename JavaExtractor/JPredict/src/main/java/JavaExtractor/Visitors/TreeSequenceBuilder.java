package JavaExtractor.Visitors;

import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;

import java.util.ArrayList;
import java.util.List;

class TreeSequenceBuilder {
    private ArrayList<Node> treeSequence;
    private ArrayList<Integer> parentIndices;
    private static String NULL_STR = "null";
    public static int NO_PARENT = -1;

    public TreeSequenceBuilder(Node node) {
        treeSequence = new ArrayList<>();
        parentIndices = new ArrayList<>();
        visitTree(node, NO_PARENT);
    }

    private void visitTree(Node node, int parentIndex) {
        if (node instanceof Comment || NULL_STR.equals(node.toString())) {
            return;
        }
        boolean isLeaf = false;
        boolean isGenericParent = isGenericParent(node);
        if (hasNoChildren(node) && isNotComment(node)) {
            if (!node.toString().isEmpty() && (!"null".equals(node.toString()) || (node instanceof NullLiteralExpr))) {
                isLeaf = true;
            }
        }
        int childId = getChildId(node);
        node.setUserData(Common.ChildId, childId);
        Property property = new Property(node, isLeaf, isGenericParent);
        node.setUserData(Common.PropertyKey, property);
        int currentIndex = treeSequence.size();
        treeSequence.add(node);
        parentIndices.add(parentIndex);
        for (Node child: node.getChildrenNodes()) {
            visitTree(child, currentIndex);
        }
    }

    ArrayList<Node> getTreeSequence() {
        return treeSequence;
    }

    ArrayList<Integer> getParentIndices() {
        return parentIndices;
    }

    private boolean isGenericParent(Node node) {
        return (node instanceof ClassOrInterfaceType)
                && ((ClassOrInterfaceType) node).getTypeArguments() != null
                && ((ClassOrInterfaceType) node).getTypeArguments().size() > 0;
    }

    private boolean hasNoChildren(Node node) {
        return node.getChildrenNodes().size() == 0;
    }

    private boolean isNotComment(Node node) {
        return !(node instanceof Comment) && !(node instanceof Statement);
    }

    private int getChildId(Node node) {
        Node parent = node.getParentNode();
        List<Node> parentsChildren = parent.getChildrenNodes();
        int childId = 0;
        for (Node child : parentsChildren) {
            if (child.getRange().equals(node.getRange())) {
                return childId;
            }
            childId++;
        }
        return childId;
    }
}
