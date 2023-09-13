/**
 * generated by Xtext 2.25.0
 */
package tau.smlab.syntech.spectra;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Define Reg Exp</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link tau.smlab.syntech.spectra.DefineRegExp#getDefineRegsList <em>Define Regs List</em>}</li>
 * </ul>
 *
 * @see tau.smlab.syntech.spectra.SpectraPackage#getDefineRegExp()
 * @model
 * @generated
 */
public interface DefineRegExp extends Decl
{
  /**
   * Returns the value of the '<em><b>Define Regs List</b></em>' containment reference list.
   * The list contents are of type {@link tau.smlab.syntech.spectra.DefineRegExpDecl}.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Define Regs List</em>' containment reference list.
   * @see tau.smlab.syntech.spectra.SpectraPackage#getDefineRegExp_DefineRegsList()
   * @model containment="true"
   * @generated
   */
  EList<DefineRegExpDecl> getDefineRegsList();

} // DefineRegExp
