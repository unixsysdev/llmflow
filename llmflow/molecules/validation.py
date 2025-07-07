"""
LLMFlow Validation Molecules

This module contains molecules for data validation and processing.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re
import logging

from ..core.base import DataAtom, ServiceAtom, ValidationResult
from ..atoms.data import StringAtom, BooleanAtom, IntegerAtom, EmailAtom
from ..atoms.service import (
    ValidateEmailAtom, ValidateStringLengthAtom, CompareStringsAtom,
    LogicAndAtom, LogicOrAtom, LogicNotAtom
)
from ..queue import QueueManager

logger = logging.getLogger(__name__)


class ValidationRequestAtom(DataAtom):
    """Data atom for validation requests."""
    
    def __init__(self, data: Dict[str, Any], rules: Dict[str, Any], 
                 metadata: Dict[str, Any] = None):
        super().__init__({'data': data, 'rules': rules}, metadata)
        self.data = data
        self.rules = rules
    
    def validate(self) -> ValidationResult:
        """Validate the validation request."""
        if not self.data:
            return ValidationResult.error("Data is required")
        
        if not self.rules:
            return ValidationResult.error("Rules are required")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize validation request to bytes."""
        import msgpack
        return msgpack.packb(self.value)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ValidationRequestAtom':
        """Deserialize bytes to validation request."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        return cls(value['data'], value['rules'])


class ValidationResultAtom(DataAtom):
    """Data atom for validation results."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, 
                 warnings: List[str] = None, field_results: Dict[str, bool] = None,
                 metadata: Dict[str, Any] = None):
        super().__init__({
            'is_valid': is_valid,
            'errors': errors or [],
            'warnings': warnings or [],
            'field_results': field_results or {},
            'validated_at': datetime.utcnow()
        }, metadata)
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.field_results = field_results or {}
    
    def validate(self) -> ValidationResult:
        """Validate the validation result."""
        return ValidationResult.success()
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
        self.value['errors'] = self.errors
        self.value['is_valid'] = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)
        self.value['warnings'] = self.warnings
    
    def serialize(self) -> bytes:
        """Serialize validation result to bytes."""
        import msgpack
        data = self.value.copy()
        data['validated_at'] = data['validated_at'].isoformat()
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ValidationResultAtom':
        """Deserialize bytes to validation result."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        return cls(
            value['is_valid'],
            value['errors'],
            value['warnings'],
            value['field_results']
        )


class DataValidationMolecule(ServiceAtom):
    """Service molecule for comprehensive data validation."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="data_validation_molecule",
            input_types=[
                "llmflow.molecules.validation.ValidationRequestAtom"
            ],
            output_types=[
                "llmflow.molecules.validation.ValidationResultAtom"
            ]
        )
        self.queue_manager = queue_manager
        
        # Initialize validation service atoms
        self.validate_email = ValidateEmailAtom()
        self.validate_string_length = ValidateStringLengthAtom()
        self.compare_strings = CompareStringsAtom()
        self.logic_and = LogicAndAtom()
        self.logic_or = LogicOrAtom()
        self.logic_not = LogicNotAtom()
        
        # Validation patterns
        self.patterns = {
            'phone': re.compile(r'^\+?1?\d{9,15}$'),
            'url': re.compile(r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'),
            'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            'credit_card': re.compile(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$')
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[ValidationResultAtom]:
        """Process data validation request."""
        validation_request = inputs[0]
        
        if not isinstance(validation_request, ValidationRequestAtom):
            return [ValidationResultAtom(False, ["Invalid validation request"])]
        
        # Validate the request itself
        request_validation = validation_request.validate()
        if not request_validation.is_valid:
            return [ValidationResultAtom(False, request_validation.errors)]
        
        # Perform validation
        result = await self._validate_data(validation_request.data, validation_request.rules)
        
        logger.info(f"Data validation completed: {result.is_valid}")
        return [result]
    
    async def _validate_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> ValidationResultAtom:
        """Validate data against rules."""
        result = ValidationResultAtom(True)
        
        for field_name, field_rules in rules.items():
            field_value = data.get(field_name)
            field_valid = await self._validate_field(field_name, field_value, field_rules)
            
            result.field_results[field_name] = field_valid.is_valid
            
            if not field_valid.is_valid:
                result.is_valid = False
                result.errors.extend(field_valid.errors)
            
            if field_valid.warnings:
                result.warnings.extend(field_valid.warnings)
        
        # Update the internal value
        result.value['is_valid'] = result.is_valid
        result.value['errors'] = result.errors
        result.value['warnings'] = result.warnings
        result.value['field_results'] = result.field_results
        
        return result
    
    async def _validate_field(self, field_name: str, field_value: Any, rules: Dict[str, Any]) -> ValidationResult:
        """Validate a single field."""
        errors = []
        warnings = []
        
        # Check if field is required
        if rules.get('required', False):
            if field_value is None or field_value == "":
                errors.append(f"{field_name} is required")
                return ValidationResult(False, errors, warnings)
        
        # Skip validation if field is empty and not required
        if field_value is None or field_value == "":
            return ValidationResult(True, errors, warnings)
        
        # Type validation
        expected_type = rules.get('type')
        if expected_type and not self._validate_type(field_value, expected_type):
            errors.append(f"{field_name} must be of type {expected_type}")
        
        # String validations
        if isinstance(field_value, str):
            # Length validation
            min_length = rules.get('min_length')
            max_length = rules.get('max_length')
            
            if min_length is not None and len(field_value) < min_length:
                errors.append(f"{field_name} must be at least {min_length} characters long")
            
            if max_length is not None and len(field_value) > max_length:
                errors.append(f"{field_name} must be no more than {max_length} characters long")
            
            # Pattern validation
            pattern = rules.get('pattern')
            if pattern:
                if pattern in self.patterns:
                    if not self.patterns[pattern].match(field_value):
                        errors.append(f"{field_name} has invalid format")
                else:
                    # Custom regex pattern
                    if not re.match(pattern, field_value):
                        errors.append(f"{field_name} does not match required pattern")
            
            # Email validation
            if rules.get('email', False):
                email_valid = self.validate_email.process([StringAtom(field_value)])[0]
                if not email_valid.value:
                    errors.append(f"{field_name} is not a valid email address")
        
        # Numeric validations
        if isinstance(field_value, (int, float)):
            min_value = rules.get('min_value')
            max_value = rules.get('max_value')
            
            if min_value is not None and field_value < min_value:
                errors.append(f"{field_name} must be at least {min_value}")
            
            if max_value is not None and field_value > max_value:
                errors.append(f"{field_name} must be no more than {max_value}")
        
        # List validations
        if isinstance(field_value, list):
            min_items = rules.get('min_items')
            max_items = rules.get('max_items')
            
            if min_items is not None and len(field_value) < min_items:
                errors.append(f"{field_name} must have at least {min_items} items")
            
            if max_items is not None and len(field_value) > max_items:
                errors.append(f"{field_name} must have no more than {max_items} items")
            
            # Validate each item if item_rules provided
            item_rules = rules.get('item_rules')
            if item_rules:
                for i, item in enumerate(field_value):
                    item_result = await self._validate_field(f"{field_name}[{i}]", item, item_rules)
                    if not item_result.is_valid:
                        errors.extend(item_result.errors)
                    warnings.extend(item_result.warnings)
        
        # Custom validation function
        custom_validator = rules.get('custom_validator')
        if custom_validator and callable(custom_validator):
            try:
                custom_result = custom_validator(field_value)
                if isinstance(custom_result, ValidationResult):
                    if not custom_result.is_valid:
                        errors.extend(custom_result.errors)
                    warnings.extend(custom_result.warnings)
                elif not custom_result:
                    errors.append(f"{field_name} failed custom validation")
            except Exception as e:
                errors.append(f"{field_name} custom validation error: {str(e)}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True


class FormValidationMolecule(ServiceAtom):
    """Service molecule for form validation."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="form_validation_molecule",
            input_types=[
                "llmflow.atoms.data.StringAtom"  # JSON form data
            ],
            output_types=[
                "llmflow.molecules.validation.ValidationResultAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.data_validation = DataValidationMolecule(queue_manager)
        
        # Common form validation rules
        self.form_rules = {
            'login_form': {
                'email': {
                    'required': True,
                    'type': 'string',
                    'email': True
                },
                'password': {
                    'required': True,
                    'type': 'string',
                    'min_length': 8
                }
            },
            'registration_form': {
                'email': {
                    'required': True,
                    'type': 'string',
                    'email': True
                },
                'password': {
                    'required': True,
                    'type': 'string',
                    'min_length': 8,
                    'pattern': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]'
                },
                'confirm_password': {
                    'required': True,
                    'type': 'string'
                },
                'first_name': {
                    'required': True,
                    'type': 'string',
                    'min_length': 2,
                    'max_length': 50
                },
                'last_name': {
                    'required': True,
                    'type': 'string',
                    'min_length': 2,
                    'max_length': 50
                },
                'phone': {
                    'required': False,
                    'type': 'string',
                    'pattern': 'phone'
                }
            },
            'profile_form': {
                'first_name': {
                    'required': True,
                    'type': 'string',
                    'min_length': 2,
                    'max_length': 50
                },
                'last_name': {
                    'required': True,
                    'type': 'string',
                    'min_length': 2,
                    'max_length': 50
                },
                'bio': {
                    'required': False,
                    'type': 'string',
                    'max_length': 500
                },
                'website': {
                    'required': False,
                    'type': 'string',
                    'pattern': 'url'
                }
            }
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[ValidationResultAtom]:
        """Process form validation request."""
        form_data_atom = inputs[0]
        
        if not isinstance(form_data_atom, StringAtom):
            return [ValidationResultAtom(False, ["Invalid form data"])]
        
        try:
            import json
            form_data = json.loads(form_data_atom.value)
        except json.JSONDecodeError:
            return [ValidationResultAtom(False, ["Invalid JSON format"])]
        
        # Determine form type
        form_type = form_data.get('form_type', 'generic')
        form_fields = form_data.get('fields', {})
        
        # Get validation rules
        rules = self.form_rules.get(form_type, {})
        
        # Custom rules can be provided
        if 'validation_rules' in form_data:
            rules.update(form_data['validation_rules'])
        
        # Create validation request
        validation_request = ValidationRequestAtom(form_fields, rules)
        
        # Validate using data validation molecule
        result = await self.data_validation.process([validation_request])
        
        # Add form-specific validations
        if form_type == 'registration_form':
            result[0] = await self._validate_registration_form(form_fields, result[0])
        
        logger.info(f"Form validation completed for {form_type}: {result[0].is_valid}")
        return result
    
    async def _validate_registration_form(self, form_fields: Dict[str, Any], 
                                        result: ValidationResultAtom) -> ValidationResultAtom:
        """Add registration-specific validations."""
        # Check password confirmation
        password = form_fields.get('password')
        confirm_password = form_fields.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            result.add_error("Password confirmation does not match")
        
        # Check if email is already registered (simplified)
        email = form_fields.get('email')
        if email and await self._is_email_registered(email):
            result.add_error("Email is already registered")
        
        return result
    
    async def _is_email_registered(self, email: str) -> bool:
        """Check if email is already registered."""
        # In production, this would check a database
        # For now, return False (not registered)
        return False


class BusinessRuleValidationMolecule(ServiceAtom):
    """Service molecule for business rule validation."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="business_rule_validation_molecule",
            input_types=[
                "llmflow.molecules.validation.ValidationRequestAtom"
            ],
            output_types=[
                "llmflow.molecules.validation.ValidationResultAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.business_rules = {}
        
        # Initialize default business rules
        self._initialize_business_rules()
    
    def _initialize_business_rules(self) -> None:
        """Initialize default business rules."""
        self.business_rules = {
            'order_validation': {
                'min_order_amount': 10.00,
                'max_order_amount': 10000.00,
                'required_fields': ['customer_id', 'items', 'total_amount'],
                'max_items_per_order': 50
            },
            'payment_validation': {
                'min_payment_amount': 0.01,
                'max_payment_amount': 50000.00,
                'required_fields': ['amount', 'currency', 'payment_method'],
                'supported_currencies': ['USD', 'EUR', 'GBP']
            },
            'inventory_validation': {
                'min_stock_level': 0,
                'max_stock_level': 100000,
                'required_fields': ['product_id', 'quantity', 'location']
            }
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[ValidationResultAtom]:
        """Process business rule validation."""
        validation_request = inputs[0]
        
        if not isinstance(validation_request, ValidationRequestAtom):
            return [ValidationResultAtom(False, ["Invalid validation request"])]
        
        # Determine business rule type
        rule_type = validation_request.rules.get('rule_type', 'generic')
        
        # Apply business rules
        result = await self._apply_business_rules(
            validation_request.data,
            rule_type,
            validation_request.rules
        )
        
        logger.info(f"Business rule validation completed for {rule_type}: {result.is_valid}")
        return [result]
    
    async def _apply_business_rules(self, data: Dict[str, Any], rule_type: str, 
                                  custom_rules: Dict[str, Any]) -> ValidationResultAtom:
        """Apply business rules to data."""
        result = ValidationResultAtom(True)
        
        # Get business rules for type
        rules = self.business_rules.get(rule_type, {})
        
        # Apply custom rules if provided
        if 'business_rules' in custom_rules:
            rules.update(custom_rules['business_rules'])
        
        # Apply specific rule validations
        if rule_type == 'order_validation':
            result = await self._validate_order_rules(data, rules, result)
        elif rule_type == 'payment_validation':
            result = await self._validate_payment_rules(data, rules, result)
        elif rule_type == 'inventory_validation':
            result = await self._validate_inventory_rules(data, rules, result)
        
        return result
    
    async def _validate_order_rules(self, data: Dict[str, Any], rules: Dict[str, Any], 
                                   result: ValidationResultAtom) -> ValidationResultAtom:
        """Validate order business rules."""
        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in data or data[field] is None:
                result.add_error(f"Required field missing: {field}")
        
        # Check order amount
        total_amount = data.get('total_amount', 0)
        min_amount = rules.get('min_order_amount', 0)
        max_amount = rules.get('max_order_amount', float('inf'))
        
        if total_amount < min_amount:
            result.add_error(f"Order amount must be at least {min_amount}")
        
        if total_amount > max_amount:
            result.add_error(f"Order amount cannot exceed {max_amount}")
        
        # Check number of items
        items = data.get('items', [])
        max_items = rules.get('max_items_per_order', float('inf'))
        
        if len(items) > max_items:
            result.add_error(f"Cannot have more than {max_items} items per order")
        
        return result
    
    async def _validate_payment_rules(self, data: Dict[str, Any], rules: Dict[str, Any], 
                                     result: ValidationResultAtom) -> ValidationResultAtom:
        """Validate payment business rules."""
        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in data or data[field] is None:
                result.add_error(f"Required field missing: {field}")
        
        # Check payment amount
        amount = data.get('amount', 0)
        min_amount = rules.get('min_payment_amount', 0)
        max_amount = rules.get('max_payment_amount', float('inf'))
        
        if amount < min_amount:
            result.add_error(f"Payment amount must be at least {min_amount}")
        
        if amount > max_amount:
            result.add_error(f"Payment amount cannot exceed {max_amount}")
        
        # Check currency
        currency = data.get('currency')
        supported_currencies = rules.get('supported_currencies', [])
        
        if currency and supported_currencies and currency not in supported_currencies:
            result.add_error(f"Currency {currency} is not supported")
        
        return result
    
    async def _validate_inventory_rules(self, data: Dict[str, Any], rules: Dict[str, Any], 
                                       result: ValidationResultAtom) -> ValidationResultAtom:
        """Validate inventory business rules."""
        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in data or data[field] is None:
                result.add_error(f"Required field missing: {field}")
        
        # Check stock levels
        quantity = data.get('quantity', 0)
        min_stock = rules.get('min_stock_level', 0)
        max_stock = rules.get('max_stock_level', float('inf'))
        
        if quantity < min_stock:
            result.add_error(f"Stock quantity must be at least {min_stock}")
        
        if quantity > max_stock:
            result.add_error(f"Stock quantity cannot exceed {max_stock}")
        
        return result
    
    async def add_business_rule(self, rule_type: str, rule_name: str, rule_value: Any) -> bool:
        """Add a new business rule."""
        if rule_type not in self.business_rules:
            self.business_rules[rule_type] = {}
        
        self.business_rules[rule_type][rule_name] = rule_value
        logger.info(f"Added business rule: {rule_type}.{rule_name} = {rule_value}")
        return True
    
    async def remove_business_rule(self, rule_type: str, rule_name: str) -> bool:
        """Remove a business rule."""
        if rule_type in self.business_rules and rule_name in self.business_rules[rule_type]:
            del self.business_rules[rule_type][rule_name]
            logger.info(f"Removed business rule: {rule_type}.{rule_name}")
            return True
        
        return False
