import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, BookOpen, Pencil, Sparkles } from "lucide-react";
import heroBg from "@/assets/hero-bg.jpg";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden hero-gradient-bg">
      {/* Background image with overlay */}
      <div className="absolute inset-0">
        <img
          src={heroBg}
          alt=""
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-background/80 to-background" />
      </div>

      {/* Decorative orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px] animate-pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/10 rounded-full blur-[100px] animate-pulse-glow" style={{ animationDelay: "1.5s" }} />

      <div className="relative z-10 container mx-auto px-6 pt-32 pb-20">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium mb-8"
          >
            <Sparkles className="h-4 w-4" />
            AI-Powered Facial Proportion Analysis
          </motion.div>

          {/* Main heading */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black leading-[0.95] tracking-tight mb-8"
          >
            Master the{" "}
            <span className="gradient-text">Loomis</span>
            <br />
            Grid Method
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-16 leading-relaxed"
          >
            Analyze facial proportions with AI precision. Learn classical construction 
            techniques or draw with real-time grid overlays that adapt to any angle.
          </motion.p>

          {/* Pathway Cards */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="grid md:grid-cols-2 gap-6 max-w-3xl mx-auto"
          >
            <PathwayCard
              to="/learn"
              icon={BookOpen}
              title="Learn Mode"
              description="Upload a portrait and get a step-by-step Loomis grid breakdown with proportion analysis"
              gradient="from-primary/20 to-primary/5"
              iconColor="text-primary"
              delay={0}
            />
            <PathwayCard
              to="/draw"
              icon={Pencil}
              title="Draw Mode"
              description="Use your webcam as a live reference with real-time grid overlay that adapts to head rotation"
              gradient="from-accent/20 to-accent/5"
              iconColor="text-accent"
              delay={0.1}
            />
          </motion.div>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  );
};

interface PathwayCardProps {
  to: string;
  icon: React.ElementType;
  title: string;
  description: string;
  gradient: string;
  iconColor: string;
  delay: number;
}

const PathwayCard = ({ to, icon: Icon, title, description, gradient, iconColor, delay }: PathwayCardProps) => (
  <Link to={to}>
    <motion.div
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ duration: 0.3 }}
      className="glass-card-hover group cursor-pointer p-8 text-left"
    >
      <div className={`inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br ${gradient} mb-6`}>
        <Icon className={`h-7 w-7 ${iconColor}`} />
      </div>
      <h3 className="text-2xl font-bold mb-3">{title}</h3>
      <p className="text-muted-foreground leading-relaxed mb-6">{description}</p>
      <div className="flex items-center gap-2 text-sm font-semibold text-primary group-hover:gap-3 transition-all duration-300">
        Get Started <ArrowRight className="h-4 w-4" />
      </div>
    </motion.div>
  </Link>
);

export default HeroSection;
