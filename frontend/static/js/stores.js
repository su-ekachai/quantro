/**
 * Alpine.js Stores for Quantro Trading Platform
 * Authentication, Theme, Internationalization, and Toast Management
 */

document.addEventListener('alpine:init', () => {
    // Authentication Store
    Alpine.store('auth', {
        token: localStorage.getItem('auth_token'),
        user: null,
        isAuthenticated: false,
        loading: false,

        init() {
            this.isAuthenticated = !!this.token;
            if (this.token) {
                this.validateToken();
            }
        },

        async login(email, password) {
            this.loading = true;

            try {
                const response = await fetch('/api/v1/auth/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        email: email,
                        password: password
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.token = data.access_token;
                    this.user = data.user || { email };
                    this.isAuthenticated = true;

                    localStorage.setItem('auth_token', this.token);

                    Alpine.store('toast').show('Login successful!', 'success');

                    // Redirect to dashboard
                    setTimeout(() => {
                        window.location.href = '/dashboard';
                    }, 1000);

                    return { success: true };
                } else {
                    let errorMessage = 'Invalid username or password';

                    try {
                        const errorData = await response.json();
                        console.log('Login error response:', errorData); // Debug log

                        if (errorData.detail) {
                            if (typeof errorData.detail === 'string') {
                                errorMessage = errorData.detail;
                            } else if (Array.isArray(errorData.detail)) {
                                // Handle validation errors
                                errorMessage = errorData.detail.map(err => err.msg || err.message || 'Validation error').join(', ');
                            } else {
                                errorMessage = 'Login failed. Please check your credentials.';
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing login response:', e);
                        errorMessage = `Login failed (${response.status}). Please try again.`;
                    }

                    Alpine.store('toast').show(errorMessage, 'error');
                    return { success: false, error: errorMessage };
                }
            } catch (error) {
                console.error('Login error:', error);
                const errorMessage = 'Network error. Please try again.';
                Alpine.store('toast').show(errorMessage, 'error');
                return { success: false, error: errorMessage };
            } finally {
                this.loading = false;
            }
        },

        async logout() {
            try {
                // Call logout endpoint if available
                if (this.token) {
                    await this.apiCall('/api/v1/auth/logout', { method: 'POST' });
                }
            } catch (error) {
                console.error('Logout error:', error);
            } finally {
                // Clear local state regardless of API call result
                this.token = null;
                this.user = null;
                this.isAuthenticated = false;

                localStorage.removeItem('auth_token');

                Alpine.store('toast').show('Logged out successfully', 'info');

                // Redirect to login
                setTimeout(() => {
                    window.location.href = '/login';
                }, 1000);
            }
        },

        async validateToken() {
            if (!this.token) {
                this.isAuthenticated = false;
                return false;
            }

            try {
                const response = await this.apiCall('/api/v1/auth/me');
                if (response && response.ok) {
                    const userData = await response.json();
                    this.user = userData;
                    this.isAuthenticated = true;
                    return true;
                } else {
                    this.logout();
                    return false;
                }
            } catch (error) {
                console.error('Token validation error:', error);
                this.logout();
                return false;
            }
        },

        async apiCall(url, options = {}) {
            if (!this.token) {
                this.logout();
                return null;
            }

            const headers = {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json',
                ...options.headers
            };

            try {
                const response = await fetch(url, { ...options, headers });

                if (response.status === 401) {
                    Alpine.store('toast').show('Session expired. Please login again.', 'warning');
                    this.logout();
                    return null;
                }

                return response;
            } catch (error) {
                console.error('API call error:', error);
                Alpine.store('toast').show('Network error. Please check your connection.', 'error');
                return null;
            }
        },

        requireAuth() {
            if (!this.isAuthenticated) {
                window.location.href = '/login';
                return false;
            }
            return true;
        }
    });

    // Theme Store
    Alpine.store('theme', {
        current: localStorage.getItem('theme') || 'light',

        init() {
            this.apply();
        },

        toggle() {
            this.current = this.current === 'light' ? 'dark' : 'light';
            this.apply();
            this.save();
        },

        set(theme) {
            if (['light', 'dark'].includes(theme)) {
                this.current = theme;
                this.apply();
                this.save();
            }
        },

        apply() {
            document.documentElement.classList.toggle('dark', this.current === 'dark');
        },

        save() {
            localStorage.setItem('theme', this.current);
        },

        isDark() {
            return this.current === 'dark';
        },

        isLight() {
            return this.current === 'light';
        }
    });

    // Internationalization Store
    Alpine.store('i18n', {
        currentLang: localStorage.getItem('lang') || 'en',
        translations: {},
        loading: false,

        async init() {
            await this.loadTranslations(this.currentLang);
        },

        async loadTranslations(lang) {
            if (!['en', 'th'].includes(lang)) {
                lang = 'en';
            }

            this.loading = true;

            try {
                const response = await fetch(`/static/translations/${lang}.json`);
                if (response.ok) {
                    this.translations = await response.json();
                    this.currentLang = lang;
                    this.save();

                    // Update document language
                    document.documentElement.lang = lang;
                } else {
                    console.error('Failed to load translations for', lang);
                    // Fallback to English if current language fails
                    if (lang !== 'en') {
                        await this.loadTranslations('en');
                    }
                }
            } catch (error) {
                console.error('Translation loading error:', error);
                // Use fallback translations
                this.translations = this.getFallbackTranslations();
            } finally {
                this.loading = false;
            }
        },

        async switchLanguage(lang) {
            if (lang !== this.currentLang) {
                await this.loadTranslations(lang);
                Alpine.store('toast').show(
                    this.t('language_changed'),
                    'success'
                );
            }
        },

        t(key, fallback = null) {
            return this.translations[key] || fallback || key;
        },

        save() {
            localStorage.setItem('lang', this.currentLang);
        },

        formatCurrency(amount, currency = 'USD') {
            const locale = this.currentLang === 'th' ? 'th-TH' : 'en-US';
            const currencyCode = this.currentLang === 'th' && currency === 'USD' ? 'THB' : currency;

            try {
                return new Intl.NumberFormat(locale, {
                    style: 'currency',
                    currency: currencyCode,
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }).format(amount);
            } catch (error) {
                console.error('Currency formatting error:', error);
                return `${currencyCode} ${amount.toFixed(2)}`;
            }
        },

        formatNumber(number, options = {}) {
            const locale = this.currentLang === 'th' ? 'th-TH' : 'en-US';

            try {
                return new Intl.NumberFormat(locale, options).format(number);
            } catch (error) {
                console.error('Number formatting error:', error);
                return number.toString();
            }
        },

        formatDate(date, options = {}) {
            const locale = this.currentLang === 'th' ? 'th-TH' : 'en-US';

            try {
                return new Intl.DateTimeFormat(locale, {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    ...options
                }).format(new Date(date));
            } catch (error) {
                console.error('Date formatting error:', error);
                return date.toString();
            }
        },

        getFallbackTranslations() {
            return {
                // Basic fallback translations
                dashboard: 'Dashboard',
                login: 'Login',
                logout: 'Logout',
                username: 'Username',
                password: 'Password',
                welcome: 'Welcome',
                error: 'Error',
                success: 'Success',
                loading: 'Loading...',
                language_changed: 'Language changed successfully'
            };
        }
    });

    // Toast Notification Store
    Alpine.store('toast', {
        toasts: [],
        nextId: 1,

        show(message, type = 'info', duration = 5000) {
            const toast = {
                id: this.nextId++,
                message,
                type,
                visible: true
            };

            this.toasts.push(toast);

            // Auto-remove after duration
            setTimeout(() => {
                this.remove(toast.id);
            }, duration);

            return toast.id;
        },

        remove(id) {
            const index = this.toasts.findIndex(toast => toast.id === id);
            if (index > -1) {
                this.toasts[index].visible = false;
                // Remove from array after animation
                setTimeout(() => {
                    this.toasts.splice(index, 1);
                }, 300);
            }
        },

        clear() {
            this.toasts.forEach(toast => {
                toast.visible = false;
            });

            setTimeout(() => {
                this.toasts = [];
            }, 300);
        },

        success(message, duration) {
            return this.show(message, 'success', duration);
        },

        error(message, duration) {
            return this.show(message, 'error', duration);
        },

        warning(message, duration) {
            return this.show(message, 'warning', duration);
        },

        info(message, duration) {
            return this.show(message, 'info', duration);
        }
    });
});
