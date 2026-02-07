import { Proportions } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="border-t border-border/30 bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-2">
            <Proportions className="h-5 w-5 text-primary" />
            <span className="text-sm font-semibold">
              Pro<span className="text-primary">-</span>Portion
            </span>
          </Link>
          <p className="text-xs text-muted-foreground">
            AI-Powered Loomis Grid Construction · Built for Artists
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
