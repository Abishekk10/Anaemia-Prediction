// Smooth scroll animations and interactions
document.addEventListener('DOMContentLoaded', function() {
    const chatbotToggle = document.getElementById('siteChatbotToggle');
    const chatbotPanel = document.getElementById('siteChatbotPanel');
    const chatbotForm = document.getElementById('siteChatbotForm');
    const chatbotInput = document.getElementById('siteChatbotInput');
    const chatbotMessages = document.getElementById('siteChatbotMessages');
    const chatbotChips = document.querySelectorAll('.site-chatbot-chip');

    const chatbotReplies = [
        {
            match: ['hi', 'hello', 'hey', 'hii', 'helo'],
            reply: 'Hi there. I am Nini, your little helper. Ask me anything about the website, prediction flow, reports, or sign up.'
        },
        {
            match: ['how are you', 'how r u', 'how are u'],
            reply: 'I am doing great and ready to help. Tell me what you want to know about the project or the website.'
        },
        {
            match: ['thank you', 'thanks', 'thx'],
            reply: 'You are very welcome. I am happy to help.'
        },
        {
            match: ['bye', 'goodbye', 'see you'],
            reply: 'Bye for now. Come back anytime if you want help with the site.'
        },
        {
            match: ['prediction', 'use prediction', 'how do i use prediction', 'how to predict', 'make prediction'],
            reply: 'Go to Prediction, enter hemoglobin, MCV, MCH, and MCHC, then submit the form to get a result with confidence and clinical guidance.'
        },
        {
            match: ['confidence', 'what does confidence mean', 'confidence score'],
            reply: 'Confidence shows how strongly the model supports its current prediction. Higher values mean the model sees a stronger pattern in the submitted lab values.'
        },
        {
            match: ['sign up', 'signup', 'register', 'create account'],
            reply: 'Use the Sign Up page to create an account with username, full name, email, contact number, and secure password. You can also use the Google button.'
        },
        {
            match: ['login', 'sign in', 'log in', 'account access'],
            reply: 'Use Login to access your account and continue with prediction, reports, and saved workflows.'
        },
        {
            match: ['report', 'read report', 'result', 'results page'],
            reply: 'The result page shows the prediction, confidence score, risk index, possible causes, recommendations, and precautions based on the submitted values.'
        },
        {
            match: ['pdf', 'upload pdf', 'upload report', 'lab report'],
            reply: 'You can upload a PDF on the prediction page. The app tries to extract hemoglobin, MCV, MCH, and MCHC automatically before generating the result.'
        },
        {
            match: ['risk', 'risk index', 'anaemia risk'],
            reply: 'Risk index is a rule-based severity indicator built from hemoglobin, MCV, MCH, and MCHC ranges. Higher values suggest a less normal blood profile.'
        },
        {
            match: ['cause', 'possible causes', 'why anaemia', 'why this result'],
            reply: 'Possible causes change with the submitted values. For example, low MCV can suggest microcytic or iron-deficiency patterns, while high MCV can suggest macrocytic or B12-folate related patterns.'
        },
        {
            match: ['recommendation', 'clinical recommendation', 'what should i do next', 'next step'],
            reply: 'Clinical recommendations are generated from the current lab pattern. More abnormal values usually produce stronger follow-up advice such as CBC review, iron studies, reticulocyte count, or vitamin testing.'
        },
        {
            match: ['precaution', 'important precautions', 'warning', 'urgent'],
            reply: 'Precautions become stronger when hemoglobin is low or the blood indices are far from normal. Severe or symptomatic cases should always be reviewed by a healthcare professional quickly.'
        },
        {
            match: ['dashboard', 'visualization', 'visualizations', 'charts'],
            reply: 'The Visualizations page shows the project charts and analysis graphics. It is useful for understanding data patterns beyond a single patient prediction.'
        },
        {
            match: ['google', 'google signup', 'google sign up'],
            reply: 'The Google button on Sign Up redirects to Google account sign-up. If you want real Google OAuth login inside this app, that can be added separately.'
        },
        {
            match: ['home', 'navigate', 'navigation', 'where is'],
            reply: 'Use the top navigation for Home, Prediction, and Visualizations. Login and Sign Up are on the right side of the navbar.'
        },
        {
            match: ['who are you', 'your name', 'what is your name'],
            reply: 'I am Nini, your cute anaemia AI helper for this website.'
        },
        {
            match: ['help', 'can you help me', 'what can you do'],
            reply: 'Yes. I can help with sign up, login, prediction, PDF upload, confidence, risk index, reports, recommendations, precautions, visualizations, and general site navigation.'
        }
    ];

    function addChatbotMessage(text, sender) {
        if (!chatbotMessages) return;
        const bubble = document.createElement('div');
        bubble.className = `site-chatbot-bubble site-chatbot-bubble-${sender}`;
        bubble.textContent = text;
        chatbotMessages.appendChild(bubble);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function getChatbotReply(question) {
        const normalized = question.toLowerCase();
        const match = chatbotReplies.find(item =>
            item.match.some(keyword => normalized.includes(keyword))
        );
        if (match) return match.reply;

        if (normalized.includes('anaemia') || normalized.includes('anemia') || normalized.includes('hemoglobin') || normalized.includes('mcv') || normalized.includes('mch') || normalized.includes('mchc')) {
            return 'That sounds like a lab-value or anaemia question. On this site, you can use the Prediction page to submit those values and then review the causes, confidence, recommendations, and precautions shown in the result.'
        }

        if (normalized.includes('account') || normalized.includes('password') || normalized.includes('email')) {
            return 'That sounds like an account question. You can use Sign Up to create an account, Login to access it, and the Google button for quick sign-up.'
        }

        if (normalized.includes('how') || normalized.includes('what') || normalized.includes('why') || normalized.includes('where')) {
            return 'I can help with sign up, login, prediction flow, PDF upload, confidence, risk index, reports, recommendations, precautions, visualizations, and navigation. Try asking in a short direct form like "What does confidence mean?" or "How do I upload a PDF?"';
        }

        return 'I am here to chat and help with most website questions. Try asking about sign up, login, prediction, PDF upload, reports, confidence, risk index, recommendations, precautions, visualizations, or just say hi.';
    }

    if (chatbotToggle && chatbotPanel) {
        chatbotToggle.addEventListener('click', function() {
            const isHidden = chatbotPanel.hasAttribute('hidden');
            if (isHidden) {
                chatbotPanel.removeAttribute('hidden');
                chatbotToggle.setAttribute('aria-expanded', 'true');
            } else {
                chatbotPanel.setAttribute('hidden', '');
                chatbotToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }

    if (chatbotForm && chatbotInput) {
        chatbotForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const question = chatbotInput.value.trim();
            if (!question) return;

            addChatbotMessage(question, 'user');
            chatbotInput.value = '';

            window.setTimeout(() => {
                addChatbotMessage(getChatbotReply(question), 'bot');
            }, 220);
        });
    }

    chatbotChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-chatbot-question');
            if (!question) return;
            addChatbotMessage(question, 'user');
            window.setTimeout(() => {
                addChatbotMessage(getChatbotReply(question), 'bot');
            }, 220);
        });
    });

    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // Add fade-in animation to elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    const animateElements = document.querySelectorAll('.card, .feature-card, .result-card');
    animateElements.forEach(el => {
        observer.observe(el);
    });

    // Navbar background on scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        }
    });

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        if (form.id === 'siteChatbotForm') {
            return;
        }
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = form.querySelectorAll('input[required], select[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                e.preventDefault();
                showAlert('Please fill in all required fields.', 'warning');
            }
        });
    });

    // Loading state for forms
    forms.forEach(form => {
        if (form.id === 'siteChatbotForm') {
            return;
        }

        form.addEventListener('submit', function() {
            const submitButton = form.querySelector('button[type="submit"]');
            if (!submitButton) {
                return;
            }

            if (!submitButton.dataset.originalText) {
                submitButton.dataset.originalText = submitButton.innerHTML;
            }

            // Let the native submit continue, then update the button state.
            window.setTimeout(() => {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';

                // Fallback for requests that do not navigate away.
                window.setTimeout(() => {
                    submitButton.disabled = false;
                    submitButton.innerHTML = submitButton.dataset.originalText;
                }, 4000);
            }, 0);
        });
    });

    // Confidence bar animation
    const confidenceBars = document.querySelectorAll('.confidence-fill');
    confidenceBars.forEach(bar => {
        const width = bar.style.width || '0%';
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 500);
    });

    // Card hover effects
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Alert system
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Loading spinner
function showLoading(container) {
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.id = 'loading-spinner';
    container.appendChild(spinner);
}

function hideLoading() {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.remove();
    }
}

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Calculate risk index color
function getRiskColor(riskIndex) {
    if (riskIndex < 30) return '#5CB85C';  // Green
    if (riskIndex < 60) return '#F0AD4E';  // Yellow
    return '#D9534F';  // Red
}

// Animate numbers
function animateNumber(element, target, duration = 1000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success');
    }).catch(() => {
        showAlert('Failed to copy to clipboard', 'danger');
    });
}

// Print functionality
function printSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Print Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .card { border: 1px solid #ddd; margin-bottom: 20px; padding: 20px; }
                .prediction-result { background: #f8f9fa; padding: 20px; border-radius: 8px; }
                @media print { body { margin: 0; } }
            </style>
        </head>
        <body>
            ${section.innerHTML}
        </body>
        </html>
    `);
    printWindow.document.close();
    printWindow.print();
}
