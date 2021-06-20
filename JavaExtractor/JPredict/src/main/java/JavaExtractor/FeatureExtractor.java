package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.FeaturesEntities.Property;
import JavaExtractor.Visitors.FunctionVisitor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("StringEquality")
class FeatureExtractor {
    private final static String separator = ",";
    private static final Set<String> s_ParentTypeToAddChildId = Stream
            .of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
            .collect(Collectors.toCollection(HashSet::new));
    private final CommandLineValues m_CommandLineValues;
    private final Path filePath;

    public FeatureExtractor(CommandLineValues commandLineValues, Path filePath) {
        this.m_CommandLineValues = commandLineValues;
        this.filePath = filePath;
    }

    public ArrayList<ProgramFeatures> extractFeatures(String code) {
        CompilationUnit m_CompilationUnit = parseFileWithRetries(code);
        FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);

        functionVisitor.visit(m_CompilationUnit, null);

        ArrayList<MethodContent> methods = functionVisitor.getMethodContents();

        return generateTreeFeatures(methods);
    }

    private CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String content = code;
        CompilationUnit parsed;
        try {
            parsed = JavaParser.parse(content);
        } catch (ParseProblemException e1) {
            // Wrap with a class and method
            try {
                content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
                parsed = JavaParser.parse(content);
            } catch (ParseProblemException e2) {
                // Wrap with a class only
                content = classPrefix + code + classSuffix;
                parsed = JavaParser.parse(content);
            }
        }

        return parsed;
    }

    private ArrayList<ProgramFeatures> generateTreeFeatures(ArrayList<MethodContent> methods) {
        ArrayList<ProgramFeatures> methodsFeatures = new ArrayList<>();
        for (MethodContent content : methods) {
            ProgramFeatures singleMethodFeatures = generateTreeFeaturesForFunction(content);
            methodsFeatures.add(singleMethodFeatures);
        }
        return methodsFeatures;
    }

    private ProgramFeatures generateTreeFeaturesForFunction(MethodContent methodContent) {
        String serializedTree = serializeTree(methodContent.getTreeAsSequence());
        return new ProgramFeatures(methodContent.getName(), serializedTree);
    }

    private String serializeTree(ArrayList<Node> treeAsSequence) {

        StringJoiner nodeTypesSequence = new StringJoiner(separator);
        StringJoiner nodeNamesSequence = new StringJoiner(separator);
        StringJoiner parentIndicesSequence = new StringJoiner(separator);

        for (Node currentNode: treeAsSequence) {
            String childId = Common.EmptyString;
            String parentRawType = Common.EmptyString;
            Property parentPropery = currentNode.getParentNode().getUserData(Common.PropertyKey);
            if (parentPropery != null) {
                parentRawType = parentPropery.getRawType();
            }
            if (currentNode.getChildrenNodes().size() == 0 || s_ParentTypeToAddChildId.contains(parentRawType)) {
                childId = saturateChildId(currentNode.getUserData(Common.ChildId))
                        .toString();
            }
            Property property = currentNode.getUserData(Common.PropertyKey);
            String name = property.getName();
            if (property.isLeaf() && name != null && !name.isEmpty()) {
                nodeNamesSequence.add(name);
            } else {
                nodeNamesSequence.add(Common.EmptyString);
            }

            nodeTypesSequence.add(String.format("%s%s", property.getType(true), childId));
            parentIndicesSequence.add(String.valueOf(property.getParentIndex()));
        }
        return nodeTypesSequence + " " + nodeNamesSequence + " " + parentIndicesSequence;
    }

    private Integer saturateChildId(int childId) {
        return Math.min(childId, m_CommandLineValues.MaxChildId);
    }
}
