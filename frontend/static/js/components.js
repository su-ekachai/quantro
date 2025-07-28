/**
 * Alpine.js Components for Quantro Trading Platform
 * Reusable UI components and form handlers
 */

console.log('Components.js file loaded');

// Wait for Alpine to be available
document.addEventListener('alpine:init', () => {
    console.log('Alpine init event fired - registering components');

    // Language Switcher Component (reusable)
    Alpine.data('languageSwitcher', () => ({
        open: false,

        toggle() {
            this.open = !this.open;
        },

        close() {
            this.open = false;
        },

        async switchLanguage(lang) {
            await Alpine.store('i18n').switchLanguage(lang);
            this.close();
        },

        get currentLang() {
            return Alpine.store('i18n').currentLang;
        },

        get loading() {
            return Alpine.store('i18n').loading;
        },

        get displayText() {
            return this.currentLang === 'th' ? 'ไทย' : 'EN';
        }
    }));



    // Login Form Component
    Alpine.data('loginForm', () => ({
        email: '',
        password: '',
        loading: false,
        errors: {},

        init() {
            // Check if already authenticated
            if (Alpine.store('auth').isAuthenticated) {
                window.location.href = '/dashboard';
            }
        },

        async submit() {
            if (this.loading) return;

            // Clear previous errors
            this.errors = {};

            // Validate form
            if (!this.validate()) {
                return;
            }

            this.loading = true;

            try {
                const result = await Alpine.store('auth').login(this.email, this.password);

                if (!result.success) {
                    // Ensure error is always a string
                    this.errors.general = typeof result.error === 'string' ? result.error : 'Login failed. Please try again.';
                }
            } catch (error) {
                console.error('Login form error:', error);
                this.errors.general = 'An unexpected error occurred. Please try again.';
            } finally {
                this.loading = false;
            }
        },

        validate() {
            const errors = {};

            if (!this.email.trim()) {
                errors.email = Alpine.store('i18n').t('email_required', 'Email is required');
            } else if (!this.isValidEmail(this.email)) {
                errors.email = Alpine.store('i18n').t('email_invalid', 'Please enter a valid email address');
            }

            if (!this.password.trim()) {
                errors.password = Alpine.store('i18n').t('password_required', 'Password is required');
            } else if (this.password.length < 6) {
                errors.password = Alpine.store('i18n').t('password_min_length', 'Password must be at least 6 characters');
            }

            this.errors = errors;
            return Object.keys(errors).length === 0;
        },

        clearError(field) {
            if (this.errors[field]) {
                delete this.errors[field];
            }
        },

        isValidEmail(email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(email);
        }
    }));

    // Navigation Component
    Alpine.data('navigation', () => ({
        mobileMenuOpen: false,

        init() {
            // Close mobile menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.$el.contains(e.target)) {
                    this.mobileMenuOpen = false;
                }
            });
        },

        toggleMobileMenu() {
            this.mobileMenuOpen = !this.mobileMenuOpen;
        },

        closeMobileMenu() {
            this.mobileMenuOpen = false;
        },

        async logout() {
            await Alpine.store('auth').logout();
        }
    }));

    // Toast Manager Component
    Alpine.data('toastManager', () => ({
        get toasts() {
            return Alpine.store('toast').toasts;
        },

        remove(id) {
            Alpine.store('toast').remove(id);
        }
    }));

    // Theme Toggle Component
    Alpine.data('themeToggle', () => ({
        get isDark() {
            return Alpine.store('theme').isDark();
        },

        toggle() {
            Alpine.store('theme').toggle();
        }
    }));

    // Form Validator Utility
    Alpine.data('formValidator', (rules = {}) => ({
        errors: {},
        touched: {},

        validate(field, value) {
            const fieldRules = rules[field];
            if (!fieldRules) return true;

            const errors = [];

            // Required validation
            if (fieldRules.required && (!value || value.toString().trim() === '')) {
                errors.push(fieldRules.required);
            }

            // Min length validation
            if (fieldRules.minLength && value && value.length < fieldRules.minLength.value) {
                errors.push(fieldRules.minLength.message);
            }

            // Max length validation
            if (fieldRules.maxLength && value && value.length > fieldRules.maxLength.value) {
                errors.push(fieldRules.maxLength.message);
            }

            // Email validation
            if (fieldRules.email && value) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(value)) {
                    errors.push(fieldRules.email);
                }
            }

            // Pattern validation
            if (fieldRules.pattern && value) {
                const regex = new RegExp(fieldRules.pattern.value);
                if (!regex.test(value)) {
                    errors.push(fieldRules.pattern.message);
                }
            }

            // Custom validation
            if (fieldRules.custom && value) {
                const customResult = fieldRules.custom(value);
                if (customResult !== true) {
                    errors.push(customResult);
                }
            }

            if (errors.length > 0) {
                this.errors[field] = errors[0]; // Show first error
                return false;
            } else {
                delete this.errors[field];
                return true;
            }
        },

        validateAll(data) {
            let isValid = true;

            Object.keys(rules).forEach(field => {
                const fieldValid = this.validate(field, data[field]);
                if (!fieldValid) {
                    isValid = false;
                }
            });

            return isValid;
        },

        touch(field) {
            this.touched[field] = true;
        },

        hasError(field) {
            return this.errors[field] && this.touched[field];
        },

        getError(field) {
            return this.hasError(field) ? this.errors[field] : '';
        },

        clearError(field) {
            delete this.errors[field];
        },

        clearAllErrors() {
            this.errors = {};
            this.touched = {};
        }
    }));

    // Authentication Guard Component
    Alpine.data('authGuard', () => ({
        init() {
            // Check authentication status
            if (!Alpine.store('auth').requireAuth()) {
                return;
            }

            // Validate token on page load
            Alpine.store('auth').validateToken();
        }
    }));

    // Loading Spinner Component
    Alpine.data('loadingSpinner', (size = 'md') => ({
        size,

        get classes() {
            const sizeClasses = {
                sm: 'w-4 h-4',
                md: 'w-6 h-6',
                lg: 'w-8 h-8',
                xl: 'w-12 h-12'
            };

            return `spinner ${sizeClasses[this.size] || sizeClasses.md}`;
        }
    }));

    // Modal Component
    Alpine.data('modal', (initialOpen = false) => ({
        open: initialOpen,

        init() {
            // Close modal on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.open) {
                    this.close();
                }
            });
        },

        show() {
            this.open = true;
            document.body.style.overflow = 'hidden';
        },

        close() {
            this.open = false;
            document.body.style.overflow = '';
        },

        toggle() {
            if (this.open) {
                this.close();
            } else {
                this.show();
            }
        }
    }));

    // Dropdown Component
    Alpine.data('dropdown', () => ({
        open: false,

        init() {
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.$el.contains(e.target)) {
                    this.open = false;
                }
            });
        },

        toggle() {
            this.open = !this.open;
        },

        close() {
            this.open = false;
        }
    }));

    // Copy to Clipboard Component
    Alpine.data('clipboard', (text = '') => ({
        text,
        copied: false,

        async copy(customText = null) {
            const textToCopy = customText || this.text;

            try {
                await navigator.clipboard.writeText(textToCopy);
                this.copied = true;

                Alpine.store('toast').success(
                    Alpine.store('i18n').t('copied_to_clipboard', 'Copied to clipboard!'),
                    2000
                );

                // Reset copied state after 2 seconds
                setTimeout(() => {
                    this.copied = false;
                }, 2000);

                return true;
            } catch (error) {
                console.error('Copy to clipboard failed:', error);
                Alpine.store('toast').error(
                    Alpine.store('i18n').t('copy_failed', 'Failed to copy to clipboard'),
                    3000
                );
                return false;
            }
        }
    }));

    // Auto-save Component
    Alpine.data('autoSave', (saveFunction, delay = 1000) => ({
        saving: false,
        lastSaved: null,
        saveTimeout: null,

        triggerSave() {
            // Clear existing timeout
            if (this.saveTimeout) {
                clearTimeout(this.saveTimeout);
            }

            // Set new timeout
            this.saveTimeout = setTimeout(async () => {
                await this.save();
            }, delay);
        },

        async save() {
            if (this.saving) return;

            this.saving = true;

            try {
                await saveFunction();
                this.lastSaved = new Date();
            } catch (error) {
                console.error('Auto-save error:', error);
                Alpine.store('toast').error(
                    Alpine.store('i18n').t('save_failed', 'Failed to save changes'),
                    3000
                );
            } finally {
                this.saving = false;
            }
        },

        get lastSavedText() {
            if (!this.lastSaved) return '';

            const now = new Date();
            const diff = Math.floor((now - this.lastSaved) / 1000);

            if (diff < 60) {
                return Alpine.store('i18n').t('saved_just_now', 'Saved just now');
            } else if (diff < 3600) {
                const minutes = Math.floor(diff / 60);
                return Alpine.store('i18n').t('saved_minutes_ago', `Saved ${minutes} minutes ago`);
            } else {
                return Alpine.store('i18n').formatDate(this.lastSaved, {
                    hour: '2-digit',
                    minute: '2-digit'
                });
            }
        }
    }));
});
