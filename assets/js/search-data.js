// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/resume.pdf";
          },
        },{id: "nav-notes",
          title: "notes",
          description: "various information",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/";
          },
        },{id: "post-exponential-family-discriminant-analysis",
      
        title: "exponential family discriminant analysis",
      
      description: "generalizing linear discriminant analysis beyond normally distributed data",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/efda/";
        
      },
    },{id: "post-bernstein-von-mises-theorem-amp-power-posteriors",
      
        title: "bernstein-von mises theorem &amp; power posteriors",
      
      description: "bayesian inference in model misspecification settings, visually explained",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/bayesian-misspecification/";
        
      },
    },{id: "post-what-if-we-didn-39-t-approximate-posteriors",
      
        title: "what if we didn&#39;t approximate posteriors?",
      
      description: "a toy bayesian neural network with an exact $$\beta \mid D$$",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/bnn/";
        
      },
    },{id: "post-new-website",
      
        title: "new website!",
      
      description: "never escaping jekyll :)",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/first-post/";
        
      },
    },{id: "news-created-this-website-old-blog-has-moved",
          title: 'created this website! old blog has moved.',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%6E%69%73%68.%6C%61%6B%6B%61%70%72%61%67%61%64%61@%79%61%6C%65.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/anish-lakkapragada", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/anishlk", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=UKAB_04AAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
