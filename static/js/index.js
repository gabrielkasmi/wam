window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // Modern scroll animations
    function handleScrollAnimations() {
        const sections = document.querySelectorAll('.section, .hero');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-section', 'is-visible');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        sections.forEach(section => {
            section.classList.add('fade-in-section');
            observer.observe(section);
        });
    }

    // Enhanced button interactions
    function enhanceButtons() {
        const buttons = document.querySelectorAll('.external-link.button');
        
        buttons.forEach(button => {
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-1px) scale(1.01)';
            });
            
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
            
            button.addEventListener('click', function() {
                // Add click ripple effect
                const ripple = document.createElement('span');
                ripple.classList.add('ripple');
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }

    // Enhanced carousel interactions
    function enhanceCarousel() {
        const carouselItems = document.querySelectorAll('.results-carousel .item');
        
        carouselItems.forEach(item => {
            item.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.01)';
                this.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.08)';
            });
            
            item.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = 'none';
            });
        });
    }

    // Smooth scroll for anchor links
    function enableSmoothScroll() {
        const links = document.querySelectorAll('a[href^="#"]');
        
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // Enhanced modal interactions
    function enhanceModal() {
        const modal = document.getElementById('poster-modal');
        const modalContent = modal.querySelector('.modal-content');
        
        if (modal && modalContent) {
            modal.addEventListener('click', function(e) {
                if (e.target === this) {
                    this.classList.remove('is-active');
                }
            });
        }
    }

    // Enhanced image interactions
    function enhanceImages() {
        const images = document.querySelectorAll('img:not(.modal img)');
        
        images.forEach(img => {
            img.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.01)';
                this.style.transition = 'transform 0.3s ease';
            });
            
            img.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
    }

    // Add loading animations
    function addLoadingAnimations() {
        // Animate elements on page load
        const heroContent = document.querySelector('.hero .hero-body');
        if (heroContent) {
            heroContent.style.opacity = '0';
            
            setTimeout(() => {
                heroContent.style.transition = 'opacity 1s ease-out';
                heroContent.style.opacity = '1';
            }, 100);
        }
    }

    // Initialize all enhancements
    function initModernFeatures() {
        handleScrollAnimations();
        enhanceButtons();
        enhanceCarousel();
        enableSmoothScroll();
        enhanceModal();
        enhanceImages();
        addLoadingAnimations();
    }

    // Initialize when DOM is ready
    initModernFeatures();

    // Re-initialize on window resize for responsive behavior
    window.addEventListener('resize', () => {
        setTimeout(initModernFeatures, 100);
    });
});
