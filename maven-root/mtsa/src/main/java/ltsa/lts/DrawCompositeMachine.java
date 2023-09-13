package ltsa.lts;
import javax.swing.*;
import java.awt.*;
import java.util.BitSet;

/**
 * @author placiana
 * This class is the exact same as DrawMachince except that in select() method 
 * it interprets the second argument as a transition, not a state.
 * This is bc Analyser.thestep returns an array of [s, s, ..., t]
 * and we wanted to preserve that legacy logic.
 */
public class DrawCompositeMachine extends DrawMachine {

    public DrawCompositeMachine(CompactState m, JPanel p, Font fn, Font fl, boolean dn, boolean nl, int separation,
            int arcIncrement) {
        super(m, p, fn, fl, dn, nl, separation, arcIncrement);
        // TODO Auto-generated constructor stub
    }


    public void select(int last, int actionId, String actionName) {
        lastaction = actionName;

        EventState es = this.mach.states[this.selected];
        while (es != null) {
            if (es.event == actionId) {
                lastselected = selected;
                selected = es.next;
                es = null;
            } else {
                es = es.list;
            }
        }
    }
}