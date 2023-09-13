/**
 * generated by Xtext 2.25.0
 */
package tau.smlab.syntech.spectra.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import tau.smlab.syntech.spectra.Constant;
import tau.smlab.syntech.spectra.SpectraPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link tau.smlab.syntech.spectra.impl.ConstantImpl#getBooleanValue <em>Boolean Value</em>}</li>
 *   <li>{@link tau.smlab.syntech.spectra.impl.ConstantImpl#getIntegerValue <em>Integer Value</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ConstantImpl extends TemporalExpressionImpl implements Constant
{
  /**
   * The default value of the '{@link #getBooleanValue() <em>Boolean Value</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getBooleanValue()
   * @generated
   * @ordered
   */
  protected static final String BOOLEAN_VALUE_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getBooleanValue() <em>Boolean Value</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getBooleanValue()
   * @generated
   * @ordered
   */
  protected String booleanValue = BOOLEAN_VALUE_EDEFAULT;

  /**
   * The default value of the '{@link #getIntegerValue() <em>Integer Value</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getIntegerValue()
   * @generated
   * @ordered
   */
  protected static final int INTEGER_VALUE_EDEFAULT = 0;

  /**
   * The cached value of the '{@link #getIntegerValue() <em>Integer Value</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getIntegerValue()
   * @generated
   * @ordered
   */
  protected int integerValue = INTEGER_VALUE_EDEFAULT;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected ConstantImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  protected EClass eStaticClass()
  {
    return SpectraPackage.Literals.CONSTANT;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String getBooleanValue()
  {
    return booleanValue;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setBooleanValue(String newBooleanValue)
  {
    String oldBooleanValue = booleanValue;
    booleanValue = newBooleanValue;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, SpectraPackage.CONSTANT__BOOLEAN_VALUE, oldBooleanValue, booleanValue));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public int getIntegerValue()
  {
    return integerValue;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setIntegerValue(int newIntegerValue)
  {
    int oldIntegerValue = integerValue;
    integerValue = newIntegerValue;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, SpectraPackage.CONSTANT__INTEGER_VALUE, oldIntegerValue, integerValue));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object eGet(int featureID, boolean resolve, boolean coreType)
  {
    switch (featureID)
    {
      case SpectraPackage.CONSTANT__BOOLEAN_VALUE:
        return getBooleanValue();
      case SpectraPackage.CONSTANT__INTEGER_VALUE:
        return getIntegerValue();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case SpectraPackage.CONSTANT__BOOLEAN_VALUE:
        setBooleanValue((String)newValue);
        return;
      case SpectraPackage.CONSTANT__INTEGER_VALUE:
        setIntegerValue((Integer)newValue);
        return;
    }
    super.eSet(featureID, newValue);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eUnset(int featureID)
  {
    switch (featureID)
    {
      case SpectraPackage.CONSTANT__BOOLEAN_VALUE:
        setBooleanValue(BOOLEAN_VALUE_EDEFAULT);
        return;
      case SpectraPackage.CONSTANT__INTEGER_VALUE:
        setIntegerValue(INTEGER_VALUE_EDEFAULT);
        return;
    }
    super.eUnset(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public boolean eIsSet(int featureID)
  {
    switch (featureID)
    {
      case SpectraPackage.CONSTANT__BOOLEAN_VALUE:
        return BOOLEAN_VALUE_EDEFAULT == null ? booleanValue != null : !BOOLEAN_VALUE_EDEFAULT.equals(booleanValue);
      case SpectraPackage.CONSTANT__INTEGER_VALUE:
        return integerValue != INTEGER_VALUE_EDEFAULT;
    }
    return super.eIsSet(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String toString()
  {
    if (eIsProxy()) return super.toString();

    StringBuilder result = new StringBuilder(super.toString());
    result.append(" (booleanValue: ");
    result.append(booleanValue);
    result.append(", integerValue: ");
    result.append(integerValue);
    result.append(')');
    return result.toString();
  }

} //ConstantImpl
