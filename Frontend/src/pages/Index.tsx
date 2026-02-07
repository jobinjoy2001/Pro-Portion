import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import Footer from "@/components/Footer";
import { motion } from "framer-motion";
import { Layers, Zap, Users, Download } from "lucide-react";

const features = [
  {
    icon: Layers,
    title: "6-Step Breakdown",
    description: "Progressive Loomis grid construction from bounding box to complete overlay",
  },
  {
    icon: Zap,
    title: "Real-Time Tracking",
    description: "3D adaptive grid that follows head rotation at 20-30 FPS",
  },
  {
    icon: Users,
    title: "Multi-Face Support",
    description: "Detect and analyze multiple faces with color-coded grids",
  },
  {
    icon: Download,
    title: "Export Everything",
    description: "Download individual steps or complete annotated analysis sets",
  },
];

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <HeroSection />

      {/* Features Section */}
      <section className="py-24 mesh-gradient">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-black mb-4">
              Everything you need to{" "}
              <span className="gradient-text">perfect proportions</span>
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Powered by MediaPipe's 478-landmark facial detection for precise, 
              artist-grade construction guides.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="glass-card-hover p-6 group"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
