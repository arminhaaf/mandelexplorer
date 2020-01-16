package nimra.mandelexplorer.util;

import java.util.LinkedList;
import java.util.function.Supplier;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class SimplePool<E> {

    private final LinkedList<E> elements = new LinkedList<>();

    private final Supplier<E> factory;

    public SimplePool(final Supplier<E> pFactory) {
        factory = pFactory;
    }

    public synchronized E borrow() {
        if (elements.size() > 0) {
            return elements.pop();
        } else {
            return createInstance();
        }
    }

    protected E createInstance() {
        return factory.get();
    }

    public synchronized void done(E p) {
        elements.add(p);
    }
}
