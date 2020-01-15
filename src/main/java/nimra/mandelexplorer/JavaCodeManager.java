package nimra.mandelexplorer;

import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.text.CharacterIterator;
import java.text.StringCharacterIterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class JavaCodeManager {

    private final Map<String, JavaByteObject> compiledClasses = Collections.synchronizedMap(new HashMap<>());

    private final JavaCodeClassLoader classLoader = new JavaCodeClassLoader();

    public void compile(String pClassName, String pJavaCode) throws Exception {
        final String tValidClassName = toValidClassName(pClassName);

        final JavaCompiler tCompiler = ToolProvider.getSystemJavaCompiler();

        final DiagnosticCollector<JavaFileObject> tDiagnostics = new DiagnosticCollector<>();

        final JavaByteObject tByteObject = new JavaByteObject(tValidClassName);

        final StandardJavaFileManager tStandardFileManager =
                tCompiler.getStandardFileManager(tDiagnostics, null, null);

        final JavaFileManager tFileManager = createFileManager(tStandardFileManager,
                                                               tByteObject);

        final JavaStringObject tJavaCode = new JavaStringObject(tValidClassName, pJavaCode);

        final JavaCompiler.CompilationTask task = tCompiler.getTask(null,
                                                                    tFileManager, tDiagnostics, null, null,
                                                                    Collections.singletonList(tJavaCode));

        if (!task.call()) {
            tDiagnostics.getDiagnostics().forEach(System.out::println);
        }

        compiledClasses.put(tValidClassName, tByteObject);
    }

    public <E> E newInstance(String pClassName, Class<E> pImplementation) throws Exception {
        //loading and using our compiled class
        Class<E> tClass = (Class<E>)classLoader.loadClass(toValidClassName(pClassName));
        return tClass.getDeclaredConstructor().newInstance();
    }

    private String toValidClassName(String pClassName) {
        if (pClassName.length() == 0) {
            return "_";
        }
        CharacterIterator ci = new StringCharacterIterator(pClassName);
        StringBuilder sb = new StringBuilder();
        for (char c = ci.first(); c != CharacterIterator.DONE; c = ci.next()) {
            if (c == ' ')
                c = '_';
            if (sb.length() == 0) {
                if (Character.isJavaIdentifierStart(c)) {
                    sb.append(c);
                    continue;
                } else
                    sb.append('_');
            }
            if (Character.isJavaIdentifierPart(c)) {
                sb.append(c);
            } else {
                sb.append('_');
            }
        }
        ;
        return sb.toString();
    }


    private class JavaCodeClassLoader extends ClassLoader {

        @Override
        public Class<?> findClass(String pClassName) throws ClassNotFoundException {
            //no need to search class path, we already have byte code.
            byte[] bytes = compiledClasses.get(pClassName).getBytes();
            return defineClass(pClassName, bytes, 0, bytes.length);
        }
    }


    private static JavaFileManager createFileManager(StandardJavaFileManager fileManager,
            JavaByteObject byteObject) {
        return new ForwardingJavaFileManager<>(fileManager) {
            @Override
            public JavaFileObject getJavaFileForOutput(JavaFileManager.Location location,
                    String className, JavaFileObject.Kind kind,
                    FileObject sibling) throws IOException {
                return byteObject;
            }
        };
    }

    public static class JavaStringObject extends SimpleJavaFileObject {
        private final String source;

        protected JavaStringObject(String name, String source) {
            super(URI.create("string:///" + name.replaceAll("\\.", "/") +
                             Kind.SOURCE.extension), Kind.SOURCE);
            this.source = source;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors)
                throws IOException {
            return source;
        }
    }

    public static class JavaByteObject extends SimpleJavaFileObject {
        private final ByteArrayOutputStream outputStream;

        protected JavaByteObject(String name) throws URISyntaxException {
            super(URI.create("bytes:///" + name + name.replaceAll("\\.", "/")), JavaFileObject.Kind.CLASS);
            outputStream = new ByteArrayOutputStream();
        }

        //overriding this to provide our OutputStream to which the
        // bytecode can be written.
        @Override
        public OutputStream openOutputStream() throws IOException {
            return outputStream;
        }

        public byte[] getBytes() {
            return outputStream.toByteArray();
        }
    }

}
